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

#include <limits>

namespace dlp::testing::framework {

template<typename T>
class ValueIterable : public IIterable
{
    class value_iterator : public IIterator
    {
      public:
        value_iterator(T    value,
                       bool report_size_inf = false,
                       bool is_consumed     = false)
            : m_value(value)
            , m_report_size_inf(report_size_inf)
            , m_is_consumed(is_consumed)
        {
        }

        std::unique_ptr<IIterator> clone() const override
        {
            return std::make_unique<value_iterator>(m_value, m_report_size_inf,
                                                    m_is_consumed);
        }

        std::any dereference() const override { return m_value; }

        void increment() override
        {
            if (!m_report_size_inf) {
                m_is_consumed = true;
            }
        }

        bool equals(const IIterator& other) const override
        {
            const auto* other_value =
                dynamic_cast<const value_iterator*>(&other);
            return other_value && m_value == other_value->m_value
                   && m_is_consumed == other_value->m_is_consumed;
        }

        bool has_next() const override
        {
            if (m_report_size_inf) {
                return true; // Infinite, always has next
            }
            // For finite single values, we never have "next" - we only have the
            // current value
            return false;
        }

        void reset() override { m_is_consumed = false; }

        size_t size() const override
        {
            if (m_report_size_inf)
                return std::numeric_limits<size_t>::max();
            return 1;
        }

      private:
        T    m_value;
        bool m_report_size_inf;
        bool m_is_consumed;
    };

  public:
    ValueIterable(T value, bool report_size_inf = false)
        : m_value(value)
        , m_report_size_inf(report_size_inf)
    {
    }

    TypeErasedIterator begin() const override
    {
        return TypeErasedIterator(
            std::make_unique<value_iterator>(m_value, m_report_size_inf));
    }

    TypeErasedIterator end() const override
    {
        return TypeErasedIterator(
            std::make_unique<value_iterator>(m_value, m_report_size_inf));
    }

    size_t size() const override
    {
        if (m_report_size_inf)
            return std::numeric_limits<size_t>::max();
        return 1;
    }
    bool empty() const override { return false; }

  private:
    T    m_value;
    bool m_report_size_inf;
};

} // namespace dlp::testing::framework
