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

#include "framework/range.hh"
#include "framework/value_iterable.hh"
#include "framework/vector_iterable.hh"
#include "parser.hh"

#include <memory>

namespace dlp { namespace testing { namespace utils {

    using dlp::testing::framework::MatrixLayout;
    using dlp::testing::framework::MatrixType;
    using dlp::testing::framework::TypeErasedRange;
    using dlp::testing::framework::ValueIterable;
    using dlp::testing::framework::VectorIterable;
    class YamlParser : public IParser
    {
      private:
        class Impl;
        std::unique_ptr<Impl> m_impl;
        YieldType             m_yield_type;

      public:
        YamlParser(const std::string& filename, std::string test_name);
        ~YamlParser();
        MicroTest&  getMicroTest() override;
        void        next() override;
        void        reset() override;
        size_t      getMicroTestCount() const override;
        void        setYieldType(YieldType yield_type) override;
        YieldType   getYieldType() const override;
        std::string getTestSuiteName() const;
        std::string getCurrentTestName() const;

        // Additional convenience methods for backward compatibility
        size_t getTestCount() const { return getMicroTestCount(); }

        // Convenience methods to access current MicroTest properties
        MatrixType   getAType() const;
        MatrixType   getBType() const;
        MatrixType   getCType() const;
        MatrixType   getAccType() const;
        MatrixLayout getStorageFormat() const;
        md_t         getM() const;
        md_t         getN() const;
        md_t         getK() const;
        md_t         getLDA() const;
        md_t         getLDB() const;
        md_t         getLDC() const;
        double       getAlpha() const;
        double       getBeta() const;
        bool         getTransA() const;
        bool         getTransB() const;
        bool         getReorderA() const;
        bool         getReorderB() const;
        bool         getPackA() const;
        bool         getPackB() const;
    };
}}} // namespace dlp::testing::utils
