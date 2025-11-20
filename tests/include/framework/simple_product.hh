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

#include "framework/cartesian_product.hh"

#include <limits>
#include <stdexcept>
#include <vector>

namespace dlp::testing::framework {

// NOTE: This can be made an iterable class.
class SimpleProduct : private CartesianProduct
{
  public:
    SimpleProduct(std::vector<TypeErasedIterator> iterators)
        : CartesianProduct(std::move(iterators))
        , m_finished(false)
    {
    }

    // Override has_next() to check our own m_finished flag
    bool has_next() const { return !m_finished; }

    std::vector<std::any> next()
    {
        // Check if we're finished
        if (m_finished) {
            throw std::runtime_error("End of simple product reached");
        }

        std::vector<std::any> result;
        for (const auto& iterator : m_iterators)
            result.push_back(iterator.dereference());

        // Try to increment all the iterators
        bool any_cant_increment = false;
        for (auto& iterator : m_iterators) {
            if (iterator.has_next()) {
                iterator.increment();
            } else {
                any_cant_increment = true;
            }
        }

        // If any iterator can't increment, we're done after this result
        if (any_cant_increment) {
            m_finished = true;
        }

        return result;
    }

    size_t size() const
    {
        // Minimum size of all the iterators.
        size_t size = std::numeric_limits<size_t>::max();
        for (const auto& iterator : m_iterators) {
            size = std::min(size, iterator.size());
        }

        if (size == std::numeric_limits<size_t>::max()) {
            // If all iterators have infinite size (size_t max), we
            // conventionally return 1 to indicate a single (degenerate) product
            // element.
            size = 1;
        }
        return size;
    }

    bool empty() const { return m_finished; }

  private:
    bool m_finished;
};

} // namespace dlp::testing::framework
