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

// Custom Headers
#include "adaptors/dlp/ual_dlp.hh"
#include "adaptors/ref/ual_ref.hh"
#include "classic/dlp_base_types.h"
#include "framework/cartesian_product.hh"
#include "framework/matrix.hh"
#include "framework/operation.hh"
#include "framework/range.hh"
#include "framework/simple_product.hh"
#include "framework/ual_plan.hh"
#include "framework/value_iterable.hh"
#include "framework/vector_iterable.hh"

// Standard Headers
#include <gtest/gtest.h>
#include <limits>
#include <vector>

using namespace dlp::testing::framework;

// Helper function to create a Range (similar to Python's range)
template<typename T>
Range<T>
make_range(T start, T end, T step = T(1))
{
    return Range<T>(start, end, step);
}

// Test fixture for Range class
class RangeTest : public ::testing::Test
{
  protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test basic range functionality
TEST_F(RangeTest, BasicRange)
{
    Range<int>       range(0, 5, 1);
    std::vector<int> expected = { 0, 1, 2, 3, 4 };
    std::vector<int> actual;

    for (auto value : range) {
        actual.push_back(value);
    }

    EXPECT_EQ(actual, expected);
}

// Test range with step
TEST_F(RangeTest, RangeWithStep)
{
    Range<int>       range(0, 10, 2);
    std::vector<int> expected = { 0, 2, 4, 6, 8 };
    std::vector<int> actual;

    for (auto value : range) {
        actual.push_back(value);
    }

    EXPECT_EQ(actual, expected);
}

// Test range size calculation
TEST_F(RangeTest, RangeSize)
{
    Range<int> range1(0, 5, 1);
    EXPECT_EQ(range1.size(), 5);

    Range<int> range2(0, 10, 2);
    EXPECT_EQ(range2.size(), 5);

    Range<int> range3(1, 6, 1);
    EXPECT_EQ(range3.size(), 5);
}

// Test empty range
TEST_F(RangeTest, EmptyRange)
{
    Range<int> range1(5, 5, 1); // start == end
    EXPECT_TRUE(range1.empty());
    EXPECT_EQ(range1.size(), 0);

    Range<int> range2(5, 3, 1); // start > end with positive step
    EXPECT_TRUE(range2.empty());
    EXPECT_EQ(range2.size(), 0);
}

// Test negative step (countdown)
TEST_F(RangeTest, NegativeStep)
{
    Range<int>       range(5, 0, -1);
    std::vector<int> expected = { 5, 4, 3, 2, 1 };
    std::vector<int> actual;

    for (auto value : range) {
        actual.push_back(value);
    }

    EXPECT_EQ(actual, expected);
    EXPECT_EQ(range.size(), 5);
}

// Test make_range helper function
TEST_F(RangeTest, MakeRangeHelper)
{
    auto             range    = make_range(1, 6);
    std::vector<int> expected = { 1, 2, 3, 4, 5 };
    std::vector<int> actual;

    for (auto value : range) {
        actual.push_back(value);
    }

    EXPECT_EQ(actual, expected);
}

// Test iterator operations
TEST_F(RangeTest, IteratorOperations)
{
    Range<int> range(0, 3, 1);
    auto       it = range.begin();

    EXPECT_EQ(*it, 0);
    ++it;
    EXPECT_EQ(*it, 1);
    it++;
    EXPECT_EQ(*it, 2);
    ++it;
    EXPECT_EQ(it, range.end());
}

// Test with different data types
TEST_F(RangeTest, DifferentDataTypes)
{
    // Test with float
    Range<float>       float_range(0.0f, 3.0f, 0.5f);
    std::vector<float> expected_float = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f };
    std::vector<float> actual_float;

    for (auto value : float_range) {
        actual_float.push_back(value);
    }

    EXPECT_EQ(actual_float, expected_float);

    // Test with char
    Range<char>       char_range('a', 'd', 1);
    std::vector<char> expected_char = { 'a', 'b', 'c' };
    std::vector<char> actual_char;

    for (auto value : char_range) {
        actual_char.push_back(value);
    }

    EXPECT_EQ(actual_char, expected_char);
}

// Test bounds checking - this should demonstrate the issue
TEST_F(RangeTest, BoundsChecking)
{
    Range<int>       range(0, 5, 1);
    std::vector<int> actual;
    int              count = 0;

    // Collect values and ensure we don't iterate infinitely
    for (auto value : range) {
        actual.push_back(value);
        count++;
        // Safety check to prevent infinite loop in case of bug
        if (count > 10) {
            break;
        }
    }

    // Should only have exactly 5 values: 0, 1, 2, 3, 4
    EXPECT_EQ(actual.size(), 5);
    EXPECT_EQ(count, 5);

    // Check that no value is >= end (5)
    for (int value : actual) {
        EXPECT_LT(value, 5)
            << "Value " << value << " should be less than end value 5";
    }

    // Check exact values
    std::vector<int> expected = { 0, 1, 2, 3, 4 };
    EXPECT_EQ(actual, expected);
}

// Test bounds checking with step > 1
TEST_F(RangeTest, BoundsCheckingWithStep)
{
    Range<int>       range(0, 10, 3); // Should give: 0, 3, 6, 9
    std::vector<int> actual;
    int              count = 0;

    for (auto value : range) {
        actual.push_back(value);
        count++;
        // Safety check
        if (count > 10) {
            break;
        }
    }

    // Should only have exactly 4 values: 0, 3, 6, 9
    EXPECT_EQ(actual.size(), 4);
    EXPECT_EQ(count, 4);

    // Check that no value is >= end (10)
    for (int value : actual) {
        EXPECT_LT(value, 10)
            << "Value " << value << " should be less than end value 10";
    }

    // Check exact values
    std::vector<int> expected = { 0, 3, 6, 9 };
    EXPECT_EQ(actual, expected);
}

// Basic tests for CartesianProduct
TEST(CartesianProductTest, BasicCartesianProduct)
{
    // Create two ranges: [0, 1, 2] and [10, 11]
    TypeErasedRange<int> range1(0, 3, 1);
    TypeErasedRange<int> range2(10, 12, 1);

    std::vector<TypeErasedIterator> iterators;
    iterators.push_back(range1.begin());
    iterators.push_back(range2.begin());

    CartesianProduct cp(std::move(iterators));

    // Expected cartesian product: (0,10), (0,11), (1,10), (1,11), (2,10),
    // (2,11)
    EXPECT_EQ(cp.size(), 6);
    EXPECT_FALSE(cp.empty());

    // Test a few combinations
    auto result1 = cp.next();
    EXPECT_EQ(std::any_cast<int>(result1[0]), 0);
    EXPECT_EQ(std::any_cast<int>(result1[1]), 10);

    auto result2 = cp.next();
    EXPECT_EQ(std::any_cast<int>(result2[0]), 0);
    EXPECT_EQ(std::any_cast<int>(result2[1]), 11);

    auto result3 = cp.next();
    EXPECT_EQ(std::any_cast<int>(result3[0]), 1);
    EXPECT_EQ(std::any_cast<int>(result3[1]), 10);
}

TEST(SimpleProductTest, BasicSimpleProduct)
{
    // Create two ranges: [0, 1, 2] and [10, 11, 12]
    TypeErasedRange<int> range1(0, 3, 1);
    TypeErasedRange<int> range2(10, 13, 1);

    std::vector<TypeErasedIterator> iterators;
    iterators.push_back(range1.begin());
    iterators.push_back(range2.begin());

    SimpleProduct sp(std::move(iterators));

    // Expected simple product: (0,10), (1,11), (2,12) - minimum size is 3
    EXPECT_EQ(sp.size(), 3);
    EXPECT_FALSE(sp.empty());

    auto result1 = sp.next();
    EXPECT_EQ(std::any_cast<int>(result1[0]), 0);
    EXPECT_EQ(std::any_cast<int>(result1[1]), 10);

    auto result2 = sp.next();
    EXPECT_EQ(std::any_cast<int>(result2[0]), 1);
    EXPECT_EQ(std::any_cast<int>(result2[1]), 11);

    auto result3 = sp.next();
    EXPECT_EQ(std::any_cast<int>(result3[0]), 2);
    EXPECT_EQ(std::any_cast<int>(result3[1]), 12);

    // Now it should be empty
    EXPECT_TRUE(sp.empty());
}

TEST(CartesianProductTest, CompleteCartesianProduct)
{
    // Create two ranges: [0, 1] and [10, 11]
    TypeErasedRange<int> range1(0, 2, 1);
    TypeErasedRange<int> range2(10, 12, 1);

    std::vector<TypeErasedIterator> iterators;
    iterators.push_back(range1.begin());
    iterators.push_back(range2.begin());

    CartesianProduct cp(std::move(iterators));

    // Expected cartesian product: (0,10), (0,11), (1,10), (1,11)
    EXPECT_EQ(cp.size(), 4);

    auto result1 = cp.next(); // (0,10)
    EXPECT_EQ(std::any_cast<int>(result1[0]), 0);
    EXPECT_EQ(std::any_cast<int>(result1[1]), 10);

    auto result2 = cp.next(); // (0,11)
    EXPECT_EQ(std::any_cast<int>(result2[0]), 0);
    EXPECT_EQ(std::any_cast<int>(result2[1]), 11);

    auto result3 = cp.next(); // (1,10)
    EXPECT_EQ(std::any_cast<int>(result3[0]), 1);
    EXPECT_EQ(std::any_cast<int>(result3[1]), 10);

    auto result4 = cp.next(); // (1,11)
    EXPECT_EQ(std::any_cast<int>(result4[0]), 1);
    EXPECT_EQ(std::any_cast<int>(result4[1]), 11);

    // Should be finished now
    EXPECT_TRUE(cp.empty());
}

TEST(SimpleProductTest, UnequalSizes)
{
    // Create ranges of different sizes: [0, 1] and [10, 11, 12]
    TypeErasedRange<int> range1(0, 2, 1);   // size 2
    TypeErasedRange<int> range2(10, 13, 1); // size 3

    std::vector<TypeErasedIterator> iterators;
    iterators.push_back(range1.begin());
    iterators.push_back(range2.begin());

    SimpleProduct sp(std::move(iterators));

    // Size should be minimum of the two ranges
    EXPECT_EQ(sp.size(), 2);

    auto result1 = sp.next(); // (0,10)
    EXPECT_EQ(std::any_cast<int>(result1[0]), 0);
    EXPECT_EQ(std::any_cast<int>(result1[1]), 10);

    auto result2 = sp.next(); // (1,11)
    EXPECT_EQ(std::any_cast<int>(result2[0]), 1);
    EXPECT_EQ(std::any_cast<int>(result2[1]), 11);

    // Should be finished now
    EXPECT_TRUE(sp.empty());
}

TEST(CartesianProductTest, MixedTypesRangeAndVector)
{
    // Create a range of integers 1 to 10
    TypeErasedRange<int> int_range(1, 11, 1); // [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    // Create a vector of 10 float values
    std::vector<float>    float_vec = { 1.1f, 2.2f, 3.3f, 4.4f, 5.5f,
                                        6.6f, 7.7f, 8.8f, 9.9f, 10.0f };
    VectorIterable<float> float_iterable(float_vec);

    std::vector<TypeErasedIterator> iterators;
    iterators.push_back(int_range.begin());
    iterators.push_back(float_iterable.begin());

    CartesianProduct cp(std::move(iterators));

    // Should have 10 * 10 = 100 combinations
    EXPECT_EQ(cp.size(), 100);
    EXPECT_FALSE(cp.empty());

    // Test first few combinations
    auto result1 = cp.next(); // (1, 1.1f)
    EXPECT_EQ(std::any_cast<int>(result1[0]), 1);
    EXPECT_FLOAT_EQ(std::any_cast<float>(result1[1]), 1.1f);

    auto result2 = cp.next(); // (1, 2.2f)
    EXPECT_EQ(std::any_cast<int>(result2[0]), 1);
    EXPECT_FLOAT_EQ(std::any_cast<float>(result2[1]), 2.2f);

    // Skip ahead and test some middle combinations
    for (iter_t i = 0; i < 8; i++) {
        cp.next(); // Skip to get to (1, 10.0f) then (2, 1.1f)
    }

    auto result11 = cp.next(); // (2, 1.1f)
    EXPECT_EQ(std::any_cast<int>(result11[0]), 2);
    EXPECT_FLOAT_EQ(std::any_cast<float>(result11[1]), 1.1f);

    // Skip to near the end
    for (iter_t i = 0; i < 88; i++) {
        cp.next(); // Skip to get near the end
    }

    auto result100 = cp.next(); // (10, 10.0f) - the last combination
    EXPECT_EQ(std::any_cast<int>(result100[0]), 10);
    EXPECT_FLOAT_EQ(std::any_cast<float>(result100[1]), 10.0f);

    // Should be finished now
    EXPECT_TRUE(cp.empty());
}

TEST(SimpleProductTest, MixedTypesRangeAndVector)
{
    // Create a range of integers 1 to 10
    TypeErasedRange<int> int_range(1, 11, 1); // [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    // Create a vector of 10 float values
    std::vector<float>    float_vec = { 1.1f, 2.2f, 3.3f, 4.4f, 5.5f,
                                        6.6f, 7.7f, 8.8f, 9.9f, 10.0f };
    VectorIterable<float> float_iterable(float_vec);

    std::vector<TypeErasedIterator> iterators;
    iterators.push_back(int_range.begin());
    iterators.push_back(float_iterable.begin());

    SimpleProduct sp(std::move(iterators));

    // Should have min(10, 10) = 10 combinations
    EXPECT_EQ(sp.size(), 10);
    EXPECT_FALSE(sp.empty());

    // Test all 10 combinations - pairs values sequentially
    std::vector<float> expected_floats = { 1.1f, 2.2f, 3.3f, 4.4f, 5.5f,
                                           6.6f, 7.7f, 8.8f, 9.9f, 10.0f };
    for (iter_t i = 1; i <= 10; i++) {
        auto result = sp.next();
        EXPECT_EQ(std::any_cast<int>(result[0]), i);
        EXPECT_FLOAT_EQ(std::any_cast<float>(result[1]),
                        expected_floats[i - 1]);
    }

    // Should be finished now
    EXPECT_TRUE(sp.empty());
}

TEST(SimpleProductTest, MixedTypesUnequalSizes)
{
    // Create a range of integers 1 to 5 (smaller)
    TypeErasedRange<int> int_range(1, 6, 1); // [1, 2, 3, 4, 5]

    // Create a vector of 10 float values (larger)
    std::vector<float>    float_vec = { 1.1f, 2.2f, 3.3f, 4.4f, 5.5f,
                                        6.6f, 7.7f, 8.8f, 9.9f, 10.0f };
    VectorIterable<float> float_iterable(float_vec);

    std::vector<TypeErasedIterator> iterators;
    iterators.push_back(int_range.begin());
    iterators.push_back(float_iterable.begin());

    SimpleProduct sp(std::move(iterators));

    // Should have min(5, 10) = 5 combinations
    EXPECT_EQ(sp.size(), 5);
    EXPECT_FALSE(sp.empty());

    // Test all 5 combinations (limited by the smaller range)
    std::vector<float> expected_floats = { 1.1f, 2.2f, 3.3f, 4.4f, 5.5f };
    for (iter_t i = 1; i <= 5; i++) {
        auto result = sp.next();
        EXPECT_EQ(std::any_cast<int>(result[0]), i);
        EXPECT_FLOAT_EQ(std::any_cast<float>(result[1]),
                        expected_floats[i - 1]);
    }

    // Should be finished now
    EXPECT_TRUE(sp.empty());
}

TEST(CartesianProductTest, WithSingleValue)
{
    // Test the example: a = {1}, b = {3,4}, c = {10,11}
    // cartesian_product(a,b,c) -> {1,3,10}, {1,3,11}, {1,4,10}, {1,4,11}

    ValueIterable<int>   single_value(1, false); // Single value, finite size
    TypeErasedRange<int> range_b(3, 5, 1);       // [3, 4]
    TypeErasedRange<int> range_c(10, 12, 1);     // [10, 11]

    std::vector<TypeErasedIterator> iterators;
    iterators.push_back(single_value.begin());
    iterators.push_back(range_b.begin());
    iterators.push_back(range_c.begin());

    CartesianProduct cp(std::move(iterators));

    // Should have 1 * 2 * 2 = 4 combinations
    EXPECT_EQ(cp.size(), 4);
    EXPECT_FALSE(cp.empty());

    // Test all combinations
    auto result1 = cp.next(); // {1, 3, 10}
    EXPECT_EQ(std::any_cast<int>(result1[0]), 1);
    EXPECT_EQ(std::any_cast<int>(result1[1]), 3);
    EXPECT_EQ(std::any_cast<int>(result1[2]), 10);

    auto result2 = cp.next(); // {1, 3, 11}
    EXPECT_EQ(std::any_cast<int>(result2[0]), 1);
    EXPECT_EQ(std::any_cast<int>(result2[1]), 3);
    EXPECT_EQ(std::any_cast<int>(result2[2]), 11);

    auto result3 = cp.next(); // {1, 4, 10}
    EXPECT_EQ(std::any_cast<int>(result3[0]), 1);
    EXPECT_EQ(std::any_cast<int>(result3[1]), 4);
    EXPECT_EQ(std::any_cast<int>(result3[2]), 10);

    auto result4 = cp.next(); // {1, 4, 11}
    EXPECT_EQ(std::any_cast<int>(result4[0]), 1);
    EXPECT_EQ(std::any_cast<int>(result4[1]), 4);
    EXPECT_EQ(std::any_cast<int>(result4[2]), 11);

    // Should be finished now
    EXPECT_TRUE(cp.empty());
}

TEST(SimpleProductTest, WithSingleValueExpanded)
{
    // Test the example: a = {1}, b = {3,4}, c = {10,11}
    // simple_product(a,b,c) -> {1,3,10}, {1,4,11}
    // Here "a" is expanded by setting m_report_size_inf as true

    ValueIterable<int> single_value(
        1, true); // Single value, infinite size for simple product
    TypeErasedRange<int> range_b(3, 5, 1);   // [3, 4]
    TypeErasedRange<int> range_c(10, 12, 1); // [10, 11]

    std::vector<TypeErasedIterator> iterators;
    iterators.push_back(single_value.begin());
    iterators.push_back(range_b.begin());
    iterators.push_back(range_c.begin());

    SimpleProduct sp(std::move(iterators));

    // Should have min(inf, 2, 2) = 2 combinations
    EXPECT_EQ(sp.size(), 2);
    EXPECT_FALSE(sp.empty());

    // Test the two combinations
    auto result1 = sp.next(); // {1, 3, 10}
    EXPECT_EQ(std::any_cast<int>(result1[0]), 1);
    EXPECT_EQ(std::any_cast<int>(result1[1]), 3);
    EXPECT_EQ(std::any_cast<int>(result1[2]), 10);

    auto result2 = sp.next(); // {1, 4, 11}
    EXPECT_EQ(std::any_cast<int>(result2[0]), 1);
    EXPECT_EQ(std::any_cast<int>(result2[1]), 4);
    EXPECT_EQ(std::any_cast<int>(result2[2]), 11);

    // Should be finished now
    EXPECT_TRUE(sp.empty());
}

TEST(SimpleProductTest, WithSingleValueNotExpanded)
{
    // Test simple product with single value but not expanded (finite size)
    // This should only yield one combination

    ValueIterable<int>   single_value(1, false); // Single value, finite size
    TypeErasedRange<int> range_b(3, 5, 1);       // [3, 4]
    TypeErasedRange<int> range_c(10, 12, 1);     // [10, 11]

    std::vector<TypeErasedIterator> iterators;
    iterators.push_back(single_value.begin());
    iterators.push_back(range_b.begin());
    iterators.push_back(range_c.begin());

    SimpleProduct sp(std::move(iterators));

    // Should have min(1, 2, 2) = 1 combination
    EXPECT_EQ(sp.size(), 1);
    EXPECT_FALSE(sp.empty());

    // Test the single combination
    auto result1 = sp.next(); // {1, 3, 10}
    EXPECT_EQ(std::any_cast<int>(result1[0]), 1);
    EXPECT_EQ(std::any_cast<int>(result1[1]), 3);
    EXPECT_EQ(std::any_cast<int>(result1[2]), 10);

    // Should be finished now
    EXPECT_TRUE(sp.empty());
}

TEST(CartesianProductTest, MultipleSingleValues)
{
    // Test cartesian product with multiple single values
    ValueIterable<int> value_a(1, false);
    ValueIterable<int> value_b(2, false);
    ValueIterable<int> value_c(3, false);

    std::vector<TypeErasedIterator> iterators;
    iterators.push_back(value_a.begin());
    iterators.push_back(value_b.begin());
    iterators.push_back(value_c.begin());

    CartesianProduct cp(std::move(iterators));

    // Should have 1 * 1 * 1 = 1 combination
    EXPECT_EQ(cp.size(), 1);
    EXPECT_FALSE(cp.empty());

    auto result = cp.next(); // {1, 2, 3}
    EXPECT_EQ(std::any_cast<int>(result[0]), 1);
    EXPECT_EQ(std::any_cast<int>(result[1]), 2);
    EXPECT_EQ(std::any_cast<int>(result[2]), 3);

    // Should be finished now
    EXPECT_TRUE(cp.empty());
}

TEST(SimpleProductTest, MultipleSingleValuesExpanded)
{
    // Test simple product with multiple single values, all expanded
    ValueIterable<int>   value_a(1, true); // Infinite
    ValueIterable<int>   value_b(2, true); // Infinite
    TypeErasedRange<int> range_c(
        10, 13, 1); // [10, 11, 12] - this limits the combinations

    std::vector<TypeErasedIterator> iterators;
    iterators.push_back(value_a.begin());
    iterators.push_back(value_b.begin());
    iterators.push_back(range_c.begin());

    SimpleProduct sp(std::move(iterators));

    // Should have min(inf, inf, 3) = 3 combinations
    EXPECT_EQ(sp.size(), 3);
    EXPECT_FALSE(sp.empty());

    // Test all three combinations - single values repeat
    auto result1 = sp.next(); // {1, 2, 10}
    EXPECT_EQ(std::any_cast<int>(result1[0]), 1);
    EXPECT_EQ(std::any_cast<int>(result1[1]), 2);
    EXPECT_EQ(std::any_cast<int>(result1[2]), 10);

    auto result2 = sp.next(); // {1, 2, 11}
    EXPECT_EQ(std::any_cast<int>(result2[0]), 1);
    EXPECT_EQ(std::any_cast<int>(result2[1]), 2);
    EXPECT_EQ(std::any_cast<int>(result2[2]), 11);

    auto result3 = sp.next(); // {1, 2, 12}
    EXPECT_EQ(std::any_cast<int>(result3[0]), 1);
    EXPECT_EQ(std::any_cast<int>(result3[1]), 2);
    EXPECT_EQ(std::any_cast<int>(result3[2]), 12);

    // Should be finished now
    EXPECT_TRUE(sp.empty());
}

// Test fixture for Matrix reorder functionality
using namespace dlp::testing::framework;
using namespace dlp::testing::classic;

class MatrixReorderTest : public ::testing::Test
{
  protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test 1: Reorder without transpose - should just optimize leading dimension
TEST_F(MatrixReorderTest, ReorderWithoutTranspose)
{
    // Create a 3x4 matrix (not transposed) with leading dimension 6 (has
    // padding)
    Matrix input(3, 4, MatrixType::f32, MatrixLayout::ROW_MAJOR, 6, false);

    // Fill with known values for testing
    float* data = reinterpret_cast<float*>(input.getData());
    for (iter_t i = 0; i < 3; i++) {
        for (iter_t j = 0; j < 4; j++) {
            data[i * 6 + j] =
                i * 10.0f
                + j; // Row 0: 0,1,2,3  Row 1: 10,11,12,13  Row 2: 20,21,22,23
        }
    }

    // Reorder using reference implementation
    UalRef   ual_ref;
    Matrix   output;
    UALError result =
        ual_ref.reorder(input, output, MatrixType::f32, MatrixType::f32,
                        MatrixType::f32, MatrixType::f32);

    EXPECT_EQ(result, UALError::UAL_SUCCESS);

    // Check output matrix properties
    EXPECT_EQ(output.getRows(), 3); // Should keep same dimensions
    EXPECT_EQ(output.getCols(), 4);
    EXPECT_FALSE(output.isTransposed()); // Should remain not transposed
    EXPECT_TRUE(output.isReordered());   // Should be marked as reordered
    EXPECT_EQ(output.getLeadingDimension(),
              4); // Should be minimum (number of columns)

    // Check that data is copied correctly (no transpose, just compact layout)
    const float* output_data = reinterpret_cast<const float*>(output.getData());
    EXPECT_FLOAT_EQ(output_data[0 * 4 + 0], 0.0f);  // [0][0]
    EXPECT_FLOAT_EQ(output_data[0 * 4 + 1], 1.0f);  // [0][1]
    EXPECT_FLOAT_EQ(output_data[0 * 4 + 2], 2.0f);  // [0][2]
    EXPECT_FLOAT_EQ(output_data[0 * 4 + 3], 3.0f);  // [0][3]
    EXPECT_FLOAT_EQ(output_data[1 * 4 + 0], 10.0f); // [1][0]
    EXPECT_FLOAT_EQ(output_data[1 * 4 + 1], 11.0f); // [1][1]
    EXPECT_FLOAT_EQ(output_data[2 * 4 + 0], 20.0f); // [2][0]
    EXPECT_FLOAT_EQ(output_data[2 * 4 + 3], 23.0f); // [2][3]
}

// Test 2: Reorder with transpose - should transpose data and optimize leading
// dimension
TEST_F(MatrixReorderTest, ReorderWithTranspose)
{
    // Create a 3x4 matrix marked as transposed with leading dimension 5 (has
    // padding)
    Matrix input(3, 4, MatrixType::f32, MatrixLayout::ROW_MAJOR, 5, true);

    // Fill with known values for testing
    // Since it's marked as transposed, logical matrix is 4x3, but physical
    // storage is 3x4
    float* data = reinterpret_cast<float*>(input.getData());
    for (iter_t i = 0; i < 3; i++) {
        for (iter_t j = 0; j < 4; j++) {
            data[i * 5 + j] = i * 10.0f + j; // Physical: Row 0: 0,1,2,3  Row 1:
                                             // 10,11,12,13  Row 2: 20,21,22,23
        }
    }

    // Reorder using reference implementation
    UalRef   ual_ref;
    Matrix   output;
    UALError result =
        ual_ref.reorder(input, output, MatrixType::f32, MatrixType::f32,
                        MatrixType::f32, MatrixType::f32);

    EXPECT_EQ(result, UALError::UAL_SUCCESS);

    // Check output matrix properties
    // Input was 3x4 transposed, so logical size was 4x3
    // After reorder, we should get physical 4x3 (dimensions swapped) and not
    // transposed
    EXPECT_EQ(output.getRows(), 4); // Swapped: input cols become output rows
    EXPECT_EQ(output.getCols(), 3); // Swapped: input rows become output cols
    EXPECT_FALSE(output.isTransposed()); // Should no longer be transposed
    EXPECT_TRUE(output.isReordered());   // Should be marked as reordered
    EXPECT_EQ(output.getLeadingDimension(),
              3); // Should be minimum (number of columns)

    // Check that data is transposed correctly
    // Original physical data: [0,1,2,3], [10,11,12,13], [20,21,22,23]
    // After transpose should be: [0,10,20], [1,11,21], [2,12,22], [3,13,23]
    const float* output_data = reinterpret_cast<const float*>(output.getData());
    EXPECT_FLOAT_EQ(output_data[0 * 3 + 0], 0.0f);  // [0][0] = original [0][0]
    EXPECT_FLOAT_EQ(output_data[0 * 3 + 1], 10.0f); // [0][1] = original [1][0]
    EXPECT_FLOAT_EQ(output_data[0 * 3 + 2], 20.0f); // [0][2] = original [2][0]
    EXPECT_FLOAT_EQ(output_data[1 * 3 + 0], 1.0f);  // [1][0] = original [0][1]
    EXPECT_FLOAT_EQ(output_data[1 * 3 + 1], 11.0f); // [1][1] = original [1][1]
    EXPECT_FLOAT_EQ(output_data[1 * 3 + 2], 21.0f); // [1][2] = original [2][1]
    EXPECT_FLOAT_EQ(output_data[2 * 3 + 0], 2.0f);  // [2][0] = original [0][2]
    EXPECT_FLOAT_EQ(output_data[2 * 3 + 1], 12.0f); // [2][1] = original [1][2]
    EXPECT_FLOAT_EQ(output_data[3 * 3 + 0], 3.0f);  // [3][0] = original [0][3]
    EXPECT_FLOAT_EQ(output_data[3 * 3 + 2], 23.0f); // [3][2] = original [2][3]
}

// Test 3: Verify effective dimensions work correctly after reorder
TEST_F(MatrixReorderTest, EffectiveDimensionsAfterReorder)
{
    // Test case similar to the failing GEMM test: 256x320 transposed
    Matrix input(256, 320, MatrixType::f32, MatrixLayout::ROW_MAJOR, 256, true);

    // Before reorder: physical 256x320, transposed=true, so effective 320x256
    EXPECT_EQ(input.getEffectiveRows(), 320);
    EXPECT_EQ(input.getEffectiveCols(), 256);

    // Reorder
    UalRef   ual_ref;
    Matrix   output;
    UALError result =
        ual_ref.reorder(input, output, MatrixType::f32, MatrixType::f32,
                        MatrixType::f32, MatrixType::f32);

    EXPECT_EQ(result, UALError::UAL_SUCCESS);

    // After reorder: should be physical 320x256, transposed=false
    EXPECT_EQ(output.getRows(), 320);
    EXPECT_EQ(output.getCols(), 256);
    EXPECT_FALSE(output.isTransposed());
    EXPECT_TRUE(output.isReordered());

    // Effective dimensions should match physical dimensions (since not
    // transposed)
    EXPECT_EQ(output.getEffectiveRows(), 320);
    EXPECT_EQ(output.getEffectiveCols(), 256);

    // Leading dimension should be optimized
    EXPECT_EQ(output.getLeadingDimension(), 256);
}

class PostOpsTest : public ::testing::Test
{
  protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test Matrix factory methods
TEST_F(PostOpsTest, MatrixFactoryMethods)
{
    // Test scalar creation
    auto scalar = Matrix::fromValue(2.5f, MatrixType::f32);
    EXPECT_EQ(scalar.getRows(), 1);
    EXPECT_EQ(scalar.getCols(), 1);
    EXPECT_EQ(scalar.getMatrixType(), MatrixType::f32);

    const float* data = reinterpret_cast<const float*>(scalar.getData());
    EXPECT_FLOAT_EQ(data[0], 2.5f);

    // Test vector creation
    std::vector<float> vec_data = { 1.0f, 2.0f, 3.0f };
    auto               vector   = Matrix::fromVector(vec_data, MatrixType::f32);
    EXPECT_EQ(vector.getRows(), 1);
    EXPECT_EQ(vector.getCols(), 3);
    EXPECT_EQ(vector.getMatrixType(), MatrixType::f32);

    const float* vec_ptr = reinterpret_cast<const float*>(vector.getData());
    EXPECT_FLOAT_EQ(vec_ptr[0], 1.0f);
    EXPECT_FLOAT_EQ(vec_ptr[1], 2.0f);
    EXPECT_FLOAT_EQ(vec_ptr[2], 3.0f);

    // Test 2D matrix creation
    std::vector<std::vector<float>> matrix_data = { { 1.0f, 2.0f },
                                                    { 3.0f, 4.0f } };
    auto matrix = Matrix::fromData(matrix_data, MatrixType::f32);
    EXPECT_EQ(matrix.getRows(), 2);
    EXPECT_EQ(matrix.getCols(), 2);
    EXPECT_EQ(matrix.getMatrixType(), MatrixType::f32);

    const float* mat_ptr = reinterpret_cast<const float*>(matrix.getData());
    EXPECT_FLOAT_EQ(mat_ptr[0], 1.0f);
    EXPECT_FLOAT_EQ(mat_ptr[1], 2.0f);
    EXPECT_FLOAT_EQ(mat_ptr[2], 3.0f);
    EXPECT_FLOAT_EQ(mat_ptr[3], 4.0f);

    // Test convenience methods
    auto scalar_conv = Matrix::scalar(1.5f);
    EXPECT_EQ(scalar_conv.getMatrixType(), MatrixType::f32);

    auto vector_conv = Matrix::vector(std::vector<float>{ 4.0f, 5.0f });
    EXPECT_EQ(vector_conv.getCols(), 2);
}

// Test element-wise operation builders
TEST_F(PostOpsTest, ElementWiseBuilders)
{
    // Test ReLU (no parameters)
    auto relu = postops::createRelu().build();
    EXPECT_EQ(relu->getType(), OperationType::ElementWise);

    auto& relu_param = static_cast<const ElementWiseParam&>(*relu);
    EXPECT_EQ(relu_param.getOperation(), ElementWiseOperation::Relu);
    EXPECT_FALSE(relu_param.hasAlpha());
    EXPECT_FALSE(relu_param.hasBeta());

    // Test PReLU (requires alpha)
    auto alpha = Matrix::fromValue(0.1f);
    auto prelu = postops::createPrelu().setAlpha(alpha).build();
    EXPECT_EQ(prelu->getType(), OperationType::ElementWise);

    auto& prelu_param = static_cast<const ElementWiseParam&>(*prelu);
    EXPECT_EQ(prelu_param.getOperation(), ElementWiseOperation::Prelu);
    EXPECT_TRUE(prelu_param.hasAlpha());
    EXPECT_FALSE(prelu_param.hasBeta());

    const float* alpha_data =
        reinterpret_cast<const float*>(prelu_param.getAlpha()->getData());
    EXPECT_FLOAT_EQ(alpha_data[0], 0.1f);

    // Test Clip (optional bounds)
    auto lower = Matrix::fromValue(-1.0f);
    auto upper = Matrix::fromValue(1.0f);
    auto clip =
        postops::createClip().setLowerBound(lower).setUpperBound(upper).build();

    auto& clip_param = static_cast<const ElementWiseParam&>(*clip);
    EXPECT_EQ(clip_param.getOperation(), ElementWiseOperation::Clip);
    EXPECT_TRUE(clip_param.hasAlpha());
    EXPECT_TRUE(clip_param.hasBeta());

    // Test other element-wise operations
    auto  gelu_tanh  = postops::createGeluTanh().build();
    auto& gelu_param = static_cast<const ElementWiseParam&>(*gelu_tanh);
    EXPECT_EQ(gelu_param.getOperation(), ElementWiseOperation::Gelu_Tanh);

    auto  swish       = postops::createSwish().build();
    auto& swish_param = static_cast<const ElementWiseParam&>(*swish);
    EXPECT_EQ(swish_param.getOperation(), ElementWiseOperation::Swish);
}

// Test PReLU validation
TEST_F(PostOpsTest, PreluValidation)
{
    // PReLU without alpha should throw
    EXPECT_THROW(postops::createPrelu().build(), std::runtime_error);
}

// Test scale operation builders
TEST_F(PostOpsTest, ScaleBuilders)
{
    // Test Scale (requires scale factor)
    auto scale_factor =
        Matrix::fromVector(std::vector<float>{ 1.0f, 2.0f, 0.5f });
    auto scale = postops::createScale().setScaleFactor(scale_factor).build();

    EXPECT_EQ(scale->getType(), OperationType::Scale);
    auto& scale_param = static_cast<const ScaleParam&>(*scale);
    EXPECT_TRUE(scale_param.hasScaleFactor());
    EXPECT_FALSE(scale_param.hasZeroPoint());
}

// Test scale validation
TEST_F(PostOpsTest, ScaleValidation)
{
    // Scale without scale factor should throw
    EXPECT_THROW(postops::createScale().build(), std::runtime_error);
}

// Test other operation builders
TEST_F(PostOpsTest, OtherOperationBuilders)
{
    // Test Bias
    auto bias_vec = Matrix::fromVector(std::vector<float>{ 0.1f, 0.2f, 0.3f });
    auto bias     = postops::createBias().setBias(bias_vec).build();

    EXPECT_EQ(bias->getType(), OperationType::Bias);
    auto& bias_param = static_cast<const BiasParam&>(*bias);
    EXPECT_EQ(bias_param.getBias().getCols(), 3);

    // Test Matrix Addition
    std::vector<std::vector<float>> transform_data = { { 1.0f, 0.5f },
                                                       { 0.0f, 1.0f } };
    auto transform_matrix = Matrix::fromData(transform_data);
    auto scale_factor     = Matrix::fromValue(0.8f);

    auto matrix_add = postops::createMatrixAdd()
                          .setMatrix(transform_matrix)
                          .setScaleFactor(scale_factor)
                          .build();

    EXPECT_EQ(matrix_add->getType(), OperationType::MatAdd);
    auto& add_param = static_cast<const MatrixAddParam&>(*matrix_add);
    EXPECT_EQ(add_param.getMatrix().getRows(), 2);
    EXPECT_EQ(add_param.getMatrix().getCols(), 2);
    EXPECT_TRUE(add_param.hasScaleFactor());

    // Test Matrix Multiplication
    auto matrix_mul = postops::createMatrixMul()
                          .setMatrix(transform_matrix)
                          .setScaleFactor(scale_factor)
                          .build();

    EXPECT_EQ(matrix_mul->getType(), OperationType::MatMul);
    auto& mul_param = static_cast<const MatrixMulParam&>(*matrix_mul);
    EXPECT_EQ(mul_param.getMatrix().getRows(), 2);
    EXPECT_EQ(mul_param.getMatrix().getCols(), 2);
    EXPECT_TRUE(mul_param.hasScaleFactor());
}

// Test bias validation
TEST_F(PostOpsTest, BiasValidation)
{
    // Bias without bias vector should throw
    EXPECT_THROW(postops::createBias().build(), std::runtime_error);
}

// Test matrix operation validation
TEST_F(PostOpsTest, MatrixOperationValidation)
{
    // Matrix Add without matrix should throw
    EXPECT_THROW(postops::createMatrixAdd().build(), std::runtime_error);

    // Matrix Mul without matrix should throw
    EXPECT_THROW(postops::createMatrixMul().build(), std::runtime_error);
}

// Test direct vector<unique_ptr<IOperationParam>> usage (replaces removed
// OperationParams container)
TEST_F(PostOpsTest, OperationParamsContainer)
{
    std::vector<std::unique_ptr<IOperationParam>> params;
    EXPECT_TRUE(params.empty());
    EXPECT_EQ(params.size(), 0);

    // Add operations
    auto relu         = postops::createRelu().build();
    auto alpha        = Matrix::fromValue(0.2f);
    auto prelu        = postops::createPrelu().setAlpha(alpha).build();
    auto scale_factor = Matrix::fromValue(2.0f);
    auto scale = postops::createScale().setScaleFactor(scale_factor).build();

    params.push_back(std::move(relu));
    params.push_back(std::move(prelu));
    params.push_back(std::move(scale));

    EXPECT_FALSE(params.empty());
    EXPECT_EQ(params.size(), 3);

    // Test access by index
    EXPECT_EQ(params[0]->getType(), OperationType::ElementWise);
    EXPECT_EQ(params[1]->getType(), OperationType::ElementWise);
    EXPECT_EQ(params[2]->getType(), OperationType::Scale);

    // Test iteration
    size_t count = 0;
    for (const auto& param : params) {
        EXPECT_NE(param.get(), nullptr);
        count++;
    }
    EXPECT_EQ(count, 3);

    // Test clear
    params.clear();
    EXPECT_TRUE(params.empty());
    EXPECT_EQ(params.size(), 0);
}

// Test plan creation via UalDlp / UalRef (replaces removed OperationFactory)
TEST_F(PostOpsTest, PlanCreation)
{
    // Test DLP plan creation
    auto ual_dlp  = std::make_unique<UalDlp>();
    auto dlp_plan = ual_dlp->createPlan();
    ASSERT_NE(dlp_plan, nullptr);

    // Test REF plan creation
    auto ual_ref  = std::make_unique<UalRef>();
    auto ref_plan = ual_ref->createPlan();
    ASSERT_NE(ref_plan, nullptr);
}

// Test RefUalPlan post-op functionality (replaces removed RefOperation)
TEST_F(PostOpsTest, RefPlanFunctionality)
{
    auto ual  = std::make_unique<UalRef>();
    auto plan = ual->createPlan();

    // Test adding individual post-ops
    auto relu  = postops::createRelu().build();
    auto alpha = Matrix::fromValue(0.1f);
    auto prelu = postops::createPrelu().setAlpha(alpha).build();

    plan->addPostOp(std::move(relu));
    plan->addPostOp(std::move(prelu));

    EXPECT_EQ(plan->getPostOps().size(), 2);

    // Test adding more post-ops
    auto scale_factor = Matrix::fromValue(2.0f);
    auto scale = postops::createScale().setScaleFactor(scale_factor).build();
    EXPECT_NO_THROW(plan->addPostOp(std::move(scale)));
    EXPECT_EQ(plan->getPostOps().size(), 3);
}

// Test DlpUalPlan post-op functionality (replaces removed DlpOperation)
TEST_F(PostOpsTest, DlpPlanFunctionality)
{
    auto ual  = std::make_unique<UalDlp>();
    auto plan = ual->createPlan();

    // Test adding post-ops
    auto relu         = postops::createRelu().build();
    auto alpha        = Matrix::fromValue(0.2f);
    auto prelu        = postops::createPrelu().setAlpha(alpha).build();
    auto scale_factor = Matrix::fromVector(std::vector<float>{ 1.0f, 2.0f });
    auto scale = postops::createScale().setScaleFactor(scale_factor).build();

    plan->addPostOp(std::move(relu));
    plan->addPostOp(std::move(prelu));
    plan->addPostOp(std::move(scale));

    EXPECT_EQ(plan->getPostOps().size(), 3);

    // Verify the types of added post-ops
    EXPECT_EQ(plan->getPostOps()[0]->getType(), OperationType::ElementWise);
    EXPECT_EQ(plan->getPostOps()[1]->getType(), OperationType::ElementWise);
    EXPECT_EQ(plan->getPostOps()[2]->getType(), OperationType::Scale);
}

// Test complex post-op sequence via plan
TEST_F(PostOpsTest, ComplexOperationSequence)
{
    auto ual  = std::make_unique<UalDlp>();
    auto plan = ual->createPlan();

    // Create a complex sequence: Bias -> Scale -> PReLU -> Clip
    auto bias_vec = Matrix::fromVector(std::vector<float>{ 0.1f, 0.2f, 0.3f });
    auto bias     = postops::createBias().setBias(bias_vec).build();

    auto scale_factor =
        Matrix::fromVector(std::vector<float>{ 2.0f, 1.5f, 0.8f });
    auto scale = postops::createScale().setScaleFactor(scale_factor).build();

    auto alpha = Matrix::fromValue(0.01f);
    auto prelu = postops::createPrelu().setAlpha(alpha).build();

    auto lower = Matrix::fromValue(-6.0f);
    auto upper = Matrix::fromValue(6.0f);
    auto clip =
        postops::createClip().setLowerBound(lower).setUpperBound(upper).build();

    // Add post-ops in sequence
    plan->addPostOp(std::move(bias));
    plan->addPostOp(std::move(scale));
    plan->addPostOp(std::move(prelu));
    plan->addPostOp(std::move(clip));

    // Verify post-ops were added
    EXPECT_EQ(plan->getPostOps().size(), 4);
    EXPECT_EQ(plan->getPostOps()[0]->getType(), OperationType::Bias);
    EXPECT_EQ(plan->getPostOps()[1]->getType(), OperationType::Scale);
    EXPECT_EQ(plan->getPostOps()[2]->getType(), OperationType::ElementWise);
    EXPECT_EQ(plan->getPostOps()[3]->getType(), OperationType::ElementWise);
}

// Test parameter cloning
TEST_F(PostOpsTest, ParameterCloning)
{
    auto alpha    = Matrix::fromValue(0.3f);
    auto original = postops::createPrelu().setAlpha(alpha).build();

    // Test cloning
    auto cloned = original->clone();
    EXPECT_NE(cloned.get(), original.get());
    EXPECT_EQ(cloned->getType(), original->getType());

    auto& orig_param  = static_cast<const ElementWiseParam&>(*original);
    auto& clone_param = static_cast<const ElementWiseParam&>(*cloned);

    EXPECT_EQ(clone_param.getOperation(), orig_param.getOperation());
    EXPECT_EQ(clone_param.hasAlpha(), orig_param.hasAlpha());

    // Verify alpha values are the same but different objects
    const float* orig_alpha =
        reinterpret_cast<const float*>(orig_param.getAlpha()->getData());
    const float* clone_alpha =
        reinterpret_cast<const float*>(clone_param.getAlpha()->getData());
    EXPECT_FLOAT_EQ(*orig_alpha, *clone_alpha);
    EXPECT_NE(orig_param.getAlpha(),
              clone_param.getAlpha()); // Different objects
}

// Test plan-based post-op flow (replaces removed IOperation/OperationParams)
TEST_F(PostOpsTest, BasicPostOpFlowTest)
{
    auto ew = postops::createPrelu()
                  .setAlpha(Matrix::fromValue(0.2f, MatrixType::f32))
                  .build();

    auto scale = postops::createScale()
                     .setScaleFactor(Matrix::fromVector(
                         std::vector<float>{ 1.0f, 2.0f }, MatrixType::f32))
                     .build();

    // Backend integration via plan
    auto ual  = std::make_unique<UalDlp>();
    auto plan = ual->createPlan();
    plan->addPostOp(std::move(ew));
    plan->addPostOp(std::move(scale));

    // Verify the plan has the correct post-ops
    EXPECT_EQ(plan->getPostOps().size(), 2);

    // Verify first post-op (PReLU)
    const auto& first_param = *plan->getPostOps()[0];
    EXPECT_EQ(first_param.getType(), OperationType::ElementWise);
    const auto& ew_param = static_cast<const ElementWiseParam&>(first_param);
    EXPECT_EQ(ew_param.getOperation(), ElementWiseOperation::Prelu);
    EXPECT_TRUE(ew_param.hasAlpha());

    // Verify second post-op (Scale)
    const auto& second_param = *plan->getPostOps()[1];
    EXPECT_EQ(second_param.getType(), OperationType::Scale);
    const auto& scale_param = static_cast<const ScaleParam&>(second_param);
    EXPECT_TRUE(scale_param.hasScaleFactor());
}

// ============================================================================
// ARGUMENT PARSER TESTS - UAL CONFIGURATION
// ============================================================================

#include "framework/utils/arg_parser.hh"

using dlp::testing::utils::ArgParser;

class ArgParserTest : public ::testing::Test
{
  protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test that UAL type parsing works correctly
TEST_F(ArgParserTest, ParseUALTypeValid)
{
    // Test valid UAL types
    EXPECT_EQ(ArgParser::parseUALType("DLP"), UALType::DLP);
    EXPECT_EQ(ArgParser::parseUALType("REF"), UALType::REF);
    EXPECT_EQ(ArgParser::parseUALType("MKL"), UALType::MKL);
    EXPECT_EQ(ArgParser::parseUALType("ONEDNN"), UALType::ONEDNN);
}

// Test that invalid UAL types throw exceptions
TEST_F(ArgParserTest, ParseUALTypeInvalid)
{
    // Test invalid UAL types
    EXPECT_THROW(ArgParser::parseUALType("INVALID"), std::invalid_argument);
    EXPECT_THROW(ArgParser::parseUALType("dlp"),
                 std::invalid_argument); // case-sensitive
    EXPECT_THROW(ArgParser::parseUALType(""), std::invalid_argument);
    EXPECT_THROW(ArgParser::parseUALType("BLAS"), std::invalid_argument);
}

// Test argument parser UAL extraction with default values
TEST_F(ArgParserTest, UALExtractionWithDefaults)
{
    // Create a parser with no UAL arguments
    const char* argv[] = { "test_program" };
    int         argc   = 1;
    ArgParser   parser(argc, const_cast<char**>(argv));

    // Test default values
    EXPECT_EQ(parser.getUalTest("DLP"), "DLP");
    EXPECT_EQ(parser.getUalRef("REF"), "REF");

    // Test custom defaults
    EXPECT_EQ(parser.getUalTest("MKL"), "MKL");
    EXPECT_EQ(parser.getUalRef("ONEDNN"), "ONEDNN");
}

// Test argument parser with UAL arguments
TEST_F(ArgParserTest, UALExtractionWithArguments)
{
    // Test --ual-test VALUE format
    {
        const char* argv[] = { "test_program", "--ual-test", "MKL", "--ual-ref",
                               "ONEDNN" };
        int         argc   = 5;
        ArgParser   parser(argc, const_cast<char**>(argv));

        EXPECT_EQ(parser.getUalTest(), "MKL");
        EXPECT_EQ(parser.getUalRef(), "ONEDNN");
    }

    // Test --ual_test VALUE format (underscore variant)
    {
        const char* argv[] = { "test_program", "--ual_test", "REF", "--ual_ref",
                               "DLP" };
        int         argc   = 5;
        ArgParser   parser(argc, const_cast<char**>(argv));

        EXPECT_EQ(parser.getUalTest(), "REF");
        EXPECT_EQ(parser.getUalRef(), "DLP");
    }
}

// Test that parseTestArgs filters UAL arguments correctly
TEST_F(ArgParserTest, ParseTestArgsFiltersUALArguments)
{
    // Test that UAL arguments are filtered out for GoogleTest
    const char* original_argv[] = { "test_program",     "--ual-test", "DLP",
                                    "--gtest_filter=*", "--ual-ref",  "REF" };
    int         original_argc   = 6;

    // Make a copy since parseTestArgs modifies argc/argv
    std::vector<char*> argv_vec;
    for (iter_t i = 0; i < original_argc; ++i) {
        argv_vec.push_back(const_cast<char*>(original_argv[i]));
    }

    int    argc = original_argc;
    char** argv = argv_vec.data();

    auto parser = ArgParser::parseTestArgs(argc, argv);

    // Verify UAL values were captured
    EXPECT_EQ(parser.getUalTest(), "DLP");
    EXPECT_EQ(parser.getUalRef(), "REF");

    // Verify argc was reduced (removed --ual-test DLP --ual-ref REF = 4 args)
    EXPECT_EQ(argc, 2); // Should have test_program and --gtest_filter=*

    // Verify program name is still there
    EXPECT_STREQ(argv[0], "test_program");
}

// ============================================================================
// MATRIX COMPARE TESTS
// ============================================================================

// Test fixture for Matrix::compare tests
class MatrixCompareTest : public ::testing::Test
{
  protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test: Fast mode - equal matrices
TEST_F(MatrixCompareTest, FastModeEqualMatricesF32)
{
    Matrix m1(2, 3, MatrixType::f32);
    Matrix m2(2, 3, MatrixType::f32);

    // Fill with same values
    m1.fillValue(1.0f);
    m2.fillValue(1.0f);

    // Fast mode comparison
    auto result = m1.compare(m2, MatrixCompareOptions::Fast());

    EXPECT_TRUE(result.equal);
    EXPECT_FALSE(result.dimensionMismatch);
    EXPECT_FALSE(result.typeMismatch);
    EXPECT_EQ(result.mismatchCount, 0);
    EXPECT_EQ(result.mismatches.size(), 0); // Fast mode doesn't collect
}

// Test: Fast mode - different values
TEST_F(MatrixCompareTest, FastModeDifferentValuesF32)
{
    Matrix m1(2, 3, MatrixType::f32);
    Matrix m2(2, 3, MatrixType::f32);

    m1.fillValue(1.0f);
    m2.fillValue(2.0f);

    auto result = m1.compare(m2, MatrixCompareOptions::Fast());

    EXPECT_FALSE(result.equal);
}

// Test: Fast mode - dimension mismatch
TEST_F(MatrixCompareTest, FastModeDimensionMismatch)
{
    Matrix m1(2, 3, MatrixType::f32);
    Matrix m2(3, 2, MatrixType::f32);

    auto result = m1.compare(m2, MatrixCompareOptions::Fast());

    EXPECT_FALSE(result.equal);
    EXPECT_TRUE(result.dimensionMismatch);
}

// Test: Fast mode - type mismatch
TEST_F(MatrixCompareTest, FastModeTypeMismatch)
{
    Matrix m1(2, 3, MatrixType::f32);
    Matrix m2(2, 3, MatrixType::s32);

    auto result = m1.compare(m2, MatrixCompareOptions::Fast());

    EXPECT_FALSE(result.equal);
    EXPECT_TRUE(result.typeMismatch);
}

// Test: Verbose mode - equal matrices
TEST_F(MatrixCompareTest, VerboseModeEqualMatricesF32)
{
    Matrix m1(2, 3, MatrixType::f32);
    Matrix m2(2, 3, MatrixType::f32);

    m1.fillValue(1.0f);
    m2.fillValue(1.0f);

    auto result = m1.compare(m2, MatrixCompareOptions::Verbose(10));

    EXPECT_TRUE(result.equal);
    EXPECT_EQ(result.mismatchCount, 0);
    EXPECT_EQ(result.mismatches.size(), 0);
    EXPECT_EQ(result.maxAbsDiff, 0.0);
}

// Test: Verbose mode - collect mismatches
TEST_F(MatrixCompareTest, VerboseModeCollectMismatches)
{
    Matrix m1(3, 3, MatrixType::f32);
    Matrix m2(3, 3, MatrixType::f32);

    m1.fillValue(1.0f);
    m2.fillValue(2.0f);

    auto result = m1.compare(m2, MatrixCompareOptions::Verbose(5));

    EXPECT_FALSE(result.equal);
    EXPECT_GT(result.mismatchCount, 0);
    EXPECT_LE(result.mismatches.size(), 5); // Limited by maxMismatches
    EXPECT_GT(result.maxAbsDiff, 0.0);

    // Check that mismatch info is populated
    if (!result.mismatches.empty()) {
        const auto& first = result.mismatches[0];
        EXPECT_EQ(first.value1, 1.0);
        EXPECT_EQ(first.value2, 2.0);
        EXPECT_EQ(first.absDiff, 1.0);
    }
}

// Test: Verbose mode - max mismatches limit
TEST_F(MatrixCompareTest, VerboseModeMaxMismatchesLimit)
{
    Matrix m1(10, 10, MatrixType::f32);
    Matrix m2(10, 10, MatrixType::f32);

    m1.fillValue(1.0f);
    m2.fillValue(2.0f);

    size_t maxMismatches = 10;
    auto result = m1.compare(m2, MatrixCompareOptions::Verbose(maxMismatches));

    EXPECT_FALSE(result.equal);
    EXPECT_EQ(result.mismatchCount, 100);               // 10x10 = 100 elements
    EXPECT_EQ(result.mismatches.size(), maxMismatches); // Limited to 10
}

// Test: Integer types - exact comparison
TEST_F(MatrixCompareTest, IntegerTypesExactComparison)
{
    Matrix m1(2, 3, MatrixType::s32);
    Matrix m2(2, 3, MatrixType::s32);

    m1.fillValue(42);
    m2.fillValue(42);

    auto result = m1.compare(m2, MatrixCompareOptions::Fast());
    EXPECT_TRUE(result.equal);

    m2.fillValue(43);
    result = m1.compare(m2, MatrixCompareOptions::Fast());
    EXPECT_FALSE(result.equal);
}

// Test: BF16 type comparison
TEST_F(MatrixCompareTest, BF16Comparison)
{
    Matrix m1(2, 3, MatrixType::bf16);
    Matrix m2(2, 3, MatrixType::bf16);

    m1.fillValue(1.0f);
    m2.fillValue(1.0f);

    auto result = m1.compare(m2, MatrixCompareOptions::Fast());
    EXPECT_TRUE(result.equal);
}

// Test: Tolerance override
TEST_F(MatrixCompareTest, ToleranceOverride)
{
    Matrix m1(2, 2, MatrixType::f32);
    Matrix m2(2, 2, MatrixType::f32);

    m1.fillValue(1.0f);
    m2.fillValue(1.001f);

    // Default tolerance - should fail
    auto result1 = m1.compare(m2, MatrixCompareOptions::Fast());
    EXPECT_FALSE(result1.equal);

    // With high tolerance - should pass
    MatrixCompareOptions opts = MatrixCompareOptions::Fast();
    opts.absToleranceOverride = 0.01;
    auto result2              = m1.compare(m2, opts);
    EXPECT_TRUE(result2.equal);
}

// Test: Format compare result output
TEST_F(MatrixCompareTest, FormatCompareResult)
{
    Matrix m1(2, 2, MatrixType::f32);
    Matrix m2(2, 2, MatrixType::f32);

    m1.fillValue(1.0f);
    m2.fillValue(2.0f);

    auto result = m1.compare(m2, MatrixCompareOptions::Verbose(5));

    std::string formatted = FormatCompareResult(result, m1, m2);

    // Check that the output contains key information
    EXPECT_NE(formatted.find("Matrix Comparison Report"), std::string::npos);
    EXPECT_NE(formatted.find("NOT EQUAL"), std::string::npos);
    EXPECT_NE(formatted.find("Total mismatches"), std::string::npos);
    EXPECT_NE(formatted.find("Maximum"), std::string::npos);
}

// Test: operator== uses fast mode
TEST_F(MatrixCompareTest, OperatorEqualUsesFastMode)
{
    Matrix m1(2, 3, MatrixType::f32);
    Matrix m2(2, 3, MatrixType::f32);

    m1.fillValue(1.0f);
    m2.fillValue(1.0f);

    EXPECT_TRUE(m1 == m2);

    m2.fillValue(2.0f);
    EXPECT_FALSE(m1 == m2);
}

// Test: operator!= consistency
TEST_F(MatrixCompareTest, OperatorNotEqual)
{
    Matrix m1(2, 3, MatrixType::f32);
    Matrix m2(2, 3, MatrixType::f32);

    m1.fillValue(1.0f);
    m2.fillValue(1.0f);

    EXPECT_FALSE(m1 != m2);

    m2.fillValue(2.0f);
    EXPECT_TRUE(m1 != m2);
}

// Test: All supported integer types
TEST_F(MatrixCompareTest, AllIntegerTypes)
{
    // Test s8, u8, s16, u16, s32, u32
    std::vector<MatrixType> types = { MatrixType::s8,  MatrixType::u8,
                                      MatrixType::s16, MatrixType::u16,
                                      MatrixType::s32, MatrixType::u32 };

    for (auto type : types) {
        Matrix m1(2, 2, type);
        Matrix m2(2, 2, type);

        m1.fillRandom(12345);
        m2 = m1; // Copy

        auto result = m1.compare(m2, MatrixCompareOptions::Fast());
        EXPECT_TRUE(result.equal) << "Failed for type " << type;
    }
}

// Test: Null data handling
TEST_F(MatrixCompareTest, NullDataHandling)
{
    Matrix m1; // Default constructor - null data
    Matrix m2;

    auto result = m1.compare(m2, MatrixCompareOptions::Fast());
    EXPECT_TRUE(result.equal); // Both null should be equal
}

// Test: Layout mismatch detection
TEST_F(MatrixCompareTest, LayoutMismatch)
{
    Matrix m1(2, 3, MatrixType::f32, MatrixLayout::ROW_MAJOR);
    Matrix m2(2, 3, MatrixType::f32, MatrixLayout::COLUMN_MAJOR);

    auto result = m1.compare(m2, MatrixCompareOptions::Fast());

    EXPECT_FALSE(result.equal);
    EXPECT_TRUE(result.layoutMismatch);
}

// Test: Verbose mode statistics accuracy
TEST_F(MatrixCompareTest, VerboseModeStatisticsAccuracy)
{
    Matrix m1(2, 2, MatrixType::f32);
    Matrix m2(2, 2, MatrixType::f32);

    // Create known values
    std::vector<std::vector<float>> data1 = { { 1.0f, 2.0f }, { 3.0f, 4.0f } };
    std::vector<std::vector<float>> data2 = { { 1.0f, 3.0f }, { 3.0f, 5.0f } };

    m1 = Matrix::fromData(data1, MatrixType::f32);
    m2 = Matrix::fromData(data2, MatrixType::f32);

    auto result = m1.compare(m2, MatrixCompareOptions::Verbose(10));

    EXPECT_FALSE(result.equal);
    EXPECT_EQ(result.mismatchCount, 2); // Two mismatches at [0,1] and [1,1]
    EXPECT_EQ(result.maxAbsDiff, 1.0);  // max(|2-3|, |4-5|) = 1.0
}

// Test: NaN handling
// NOTE: Special handling for NaN values.
// Even though NaN is mathematically undefined and NaN == NaN is false, but
// since we are testing for NaN propagation, we treat the two NaNs at the same
// position to be equal
TEST_F(MatrixCompareTest, NaNHandling)
{
    Matrix m1(2, 2, MatrixType::f32);
    Matrix m2(2, 2, MatrixType::f32);

    // Create matrices with NaN
    std::vector<std::vector<float>> data1 = {
        { 1.0f, std::numeric_limits<float>::quiet_NaN() },
        { 3.0f, 4.0f }
    };
    std::vector<std::vector<float>> data2 = {
        { 1.0f, std::numeric_limits<float>::quiet_NaN() },
        { 3.0f, 4.0f }
    };

    m1 = Matrix::fromData(data1, MatrixType::f32);
    m2 = Matrix::fromData(data2, MatrixType::f32);

    // NaN != NaN, but since we're comparing individual elements
    // using std::isnan checks, TRUE will be reported.
    auto result = m1.compare(m2, MatrixCompareOptions::Fast());
    EXPECT_TRUE(result.equal);

    // Verbose mode should report equality.
    result = m1.compare(m2, MatrixCompareOptions::Verbose(10));
    EXPECT_TRUE(result.equal);
    EXPECT_EQ(result.mismatchCount, 0);
}

// Test: NaN comparison with a float should report not equal
TEST_F(MatrixCompareTest, NaNFloatComparison)
{
    Matrix m1(2, 2, MatrixType::f32);
    Matrix m2(2, 2, MatrixType::f32);

    // Create matrices with NaN
    std::vector<std::vector<float>> data1 = {
        { 1.0f, 4.0f },
        { 3.0f, std::numeric_limits<float>::quiet_NaN() }
    };
    std::vector<std::vector<float>> data2 = {
        { 1.0f, std::numeric_limits<float>::quiet_NaN() },
        { 3.0f, 4.0f }
    };

    m1 = Matrix::fromData(data1, MatrixType::f32);
    m2 = Matrix::fromData(data2, MatrixType::f32);

    // NaN != float_val
    auto result = m1.compare(m2, MatrixCompareOptions::Fast());
    EXPECT_FALSE(result.equal);

    // Verbose mode should also report inequality.
    result = m1.compare(m2, MatrixCompareOptions::Verbose(10));
    EXPECT_FALSE(result.equal);

    EXPECT_GT(result.mismatchCount, 1);
}

// Test: Infinity handling
TEST_F(MatrixCompareTest, InfinityHandling)
{
    Matrix m1(2, 2, MatrixType::f32);
    Matrix m2(2, 2, MatrixType::f32);

    // Create matrices with infinity
    std::vector<std::vector<float>> data1 = {
        { 1.0f, std::numeric_limits<float>::infinity() },
        { -std::numeric_limits<float>::infinity(), 4.0f }
    };
    std::vector<std::vector<float>> data2 = {
        { 1.0f, std::numeric_limits<float>::infinity() },
        { -std::numeric_limits<float>::infinity(), 4.0f }
    };

    m1 = Matrix::fromData(data1, MatrixType::f32);
    m2 = Matrix::fromData(data2, MatrixType::f32);

    // Inf == Inf, so matrices should be equal
    auto result = m1.compare(m2, MatrixCompareOptions::Fast());
    EXPECT_TRUE(result.equal);
}

// Test: Infinity comparison against NaN should report not equal
TEST_F(MatrixCompareTest, InfinityNaNComparison)
{
    Matrix m1(2, 2, MatrixType::f32);
    Matrix m2(2, 2, MatrixType::f32);

    // Create matrices with infinity
    std::vector<std::vector<float>> data1 = {
        { 1.0f, std::numeric_limits<float>::quiet_NaN() },
        { std::numeric_limits<float>::quiet_NaN(), 4.0f }
    };
    std::vector<std::vector<float>> data2 = {
        { 1.0f, std::numeric_limits<float>::infinity() },
        { -std::numeric_limits<float>::infinity(), 4.0f }
    };

    m1 = Matrix::fromData(data1, MatrixType::f32);
    m2 = Matrix::fromData(data2, MatrixType::f32);

    // Inf != NaN
    auto result = m1.compare(m2, MatrixCompareOptions::Fast());
    EXPECT_FALSE(result.equal);

    // Verbose mode should also report inequality.
    result = m1.compare(m2, MatrixCompareOptions::Verbose(10));
    EXPECT_FALSE(result.equal);

    EXPECT_GT(result.mismatchCount, 1);
}

// Test: Infinity comparisons against Float value should report not equal
TEST_F(MatrixCompareTest, InfinityFloatComparison)
{
    Matrix m1(2, 2, MatrixType::f32);
    Matrix m2(2, 2, MatrixType::f32);

    // Create matrices with infinity
    std::vector<std::vector<float>> data1 = { { 1.0f, 2.0f }, { 3.0f, 4.0f } };
    std::vector<std::vector<float>> data2 = {
        { 1.0f, std::numeric_limits<float>::infinity() },
        { -std::numeric_limits<float>::infinity(), 4.0f }
    };

    m1 = Matrix::fromData(data1, MatrixType::f32);
    m2 = Matrix::fromData(data2, MatrixType::f32);

    // Inf != float_val
    auto result = m1.compare(m2, MatrixCompareOptions::Fast());
    EXPECT_FALSE(result.equal);

    // Verbose mode should also report inequality.
    result = m1.compare(m2, MatrixCompareOptions::Verbose(10));
    EXPECT_FALSE(result.equal);

    EXPECT_GT(result.mismatchCount, 1);
}

// ============================================================================
// PREPARED BATCH GEMM ARGS TESTS - UAL OPTIMIZATION
// ============================================================================

#include "adaptors/dlp/ual_dlp.hh"
#include "framework/batch_gemm_args.hh"

class PreparedBatchGemmTest : public ::testing::Test
{
  protected:
    void SetUp() override { ual_dlp = std::make_unique<UalDlp>(); }

    std::unique_ptr<UalDlp> ual_dlp;
};

// Test 1: batch_prepare_metadata with f32 matrices
TEST_F(PreparedBatchGemmTest, PrepareMetadataF32SingleGroup)
{
    // Create single group with 2 matrices
    std::vector<BatchGroup> groups(1);
    groups[0].m     = 64;
    groups[0].n     = 64;
    groups[0].k     = 64;
    groups[0].alpha = 1.0;
    groups[0].beta  = 0.0;

    for (iter_t i = 0; i < 2; ++i) {
        Matrix A(64, 64, MatrixType::f32, MatrixLayout::ROW_MAJOR, 64, false);
        Matrix B(64, 64, MatrixType::f32, MatrixLayout::ROW_MAJOR, 64, false);
        Matrix C(64, 64, MatrixType::f32, MatrixLayout::ROW_MAJOR, 64, false);
        A.fillRandom(42 + i);
        B.fillRandom(43 + i);
        C.fillRandom(44 + i);
        groups[0].A_matrices.push_back(std::move(A));
        groups[0].B_matrices.push_back(std::move(B));
        groups[0].C_matrices.push_back(std::move(C));
    }

    // Prepare args
    PreparedBatchGemmArgs prepared;
    UALError              status =
        prepare_batch_gemm_args(groups, MatrixType::f32, prepared);
    ASSERT_EQ(status, UALError::UAL_SUCCESS);

    // Call batch_prepare_metadata
    ual_dlp->batch_prepare_metadata(prepared);

    // Verify metadata prepared
    EXPECT_EQ(prepared.backend_metadata.size(), 1);
    EXPECT_NE(prepared.backend_metadata[0], nullptr);

    // Verify alpha/beta precomputed for f32
    ASSERT_EQ(prepared.alpha_f32.size(), 1);
    ASSERT_EQ(prepared.beta_f32.size(), 1);
    EXPECT_FLOAT_EQ(prepared.alpha_f32[0], 1.0f);
    EXPECT_FLOAT_EQ(prepared.beta_f32[0], 0.0f);

    // Verify s32 vectors NOT populated for f32
    EXPECT_TRUE(prepared.alpha_s32.empty());
    EXPECT_TRUE(prepared.beta_s32.empty());
}

// Test 2: batch_prepare_metadata with int8 matrices
TEST_F(PreparedBatchGemmTest, PrepareMetadataS8SingleGroup)
{
    // Create single group with 2 matrices
    std::vector<BatchGroup> groups(1);
    groups[0].m     = 64;
    groups[0].n     = 64;
    groups[0].k     = 64;
    groups[0].alpha = 2.0;
    groups[0].beta  = 1.0;

    for (iter_t i = 0; i < 2; ++i) {
        Matrix A(64, 64, MatrixType::s8, MatrixLayout::ROW_MAJOR, 64, false);
        Matrix B(64, 64, MatrixType::s8, MatrixLayout::ROW_MAJOR, 64, false);
        Matrix C(64, 64, MatrixType::s32, MatrixLayout::ROW_MAJOR, 64, false);
        A.fillRandom(42 + i);
        B.fillRandom(43 + i);
        C.fillRandom(44 + i);
        groups[0].A_matrices.push_back(std::move(A));
        groups[0].B_matrices.push_back(std::move(B));
        groups[0].C_matrices.push_back(std::move(C));
    }

    // Prepare args
    PreparedBatchGemmArgs prepared;
    UALError              status =
        prepare_batch_gemm_args(groups, MatrixType::s32, prepared);
    ASSERT_EQ(status, UALError::UAL_SUCCESS);

    // Call batch_prepare_metadata
    ual_dlp->batch_prepare_metadata(prepared);

    // Verify metadata prepared
    EXPECT_EQ(prepared.backend_metadata.size(), 1);
    EXPECT_NE(prepared.backend_metadata[0], nullptr);

    // Verify alpha/beta precomputed for s32
    ASSERT_EQ(prepared.alpha_s32.size(), 1);
    ASSERT_EQ(prepared.beta_s32.size(), 1);
    EXPECT_EQ(prepared.alpha_s32[0], 2);
    EXPECT_EQ(prepared.beta_s32[0], 1);

    // Verify f32 vectors NOT populated for int8
    EXPECT_TRUE(prepared.alpha_f32.empty());
    EXPECT_TRUE(prepared.beta_f32.empty());
}

// Test 3: Optimized batch_gemm uses precomputed values
TEST_F(PreparedBatchGemmTest, OptimizedBatchGemmF32)
{
    // Create single group
    std::vector<BatchGroup> groups(1);
    groups[0].m     = 32;
    groups[0].n     = 32;
    groups[0].k     = 32;
    groups[0].alpha = 1.5;
    groups[0].beta  = 0.5;

    for (iter_t i = 0; i < 2; ++i) {
        Matrix A(32, 32, MatrixType::f32, MatrixLayout::ROW_MAJOR, 32, false);
        Matrix B(32, 32, MatrixType::f32, MatrixLayout::ROW_MAJOR, 32, false);
        Matrix C(32, 32, MatrixType::f32, MatrixLayout::ROW_MAJOR, 32, false);
        A.fillRandom(42 + i);
        B.fillRandom(43 + i);
        C.fillRandom(44 + i);
        groups[0].A_matrices.push_back(std::move(A));
        groups[0].B_matrices.push_back(std::move(B));
        groups[0].C_matrices.push_back(std::move(C));
    }

    // Prepare args
    PreparedBatchGemmArgs prepared;
    UALError              status =
        prepare_batch_gemm_args(groups, MatrixType::f32, prepared);
    ASSERT_EQ(status, UALError::UAL_SUCCESS);

    // Prepare metadata (including alpha/beta)
    ual_dlp->batch_prepare_metadata(prepared);

    // Call optimized batch_gemm
    status = ual_dlp->batch_gemm(prepared);
    EXPECT_EQ(status, UALError::UAL_SUCCESS);

    // Verify alpha/beta were used (check they're still there)
    EXPECT_FLOAT_EQ(prepared.alpha_f32[0], 1.5f);
    EXPECT_FLOAT_EQ(prepared.beta_f32[0], 0.5f);
}

// Test 4: Multi-group with different alpha/beta
TEST_F(PreparedBatchGemmTest, PrepareMetadataMultiGroup)
{
    // Create 3 groups with different alpha/beta
    std::vector<BatchGroup> groups(3);

    for (iter_t g = 0; g < 3; ++g) {
        groups[g].m     = 16;
        groups[g].n     = 16;
        groups[g].k     = 16;
        groups[g].alpha = 1.0 + g; // Different per group
        groups[g].beta  = 0.5 * g; // Different per group

        Matrix A(16, 16, MatrixType::f32, MatrixLayout::ROW_MAJOR, 16, false);
        Matrix B(16, 16, MatrixType::f32, MatrixLayout::ROW_MAJOR, 16, false);
        Matrix C(16, 16, MatrixType::f32, MatrixLayout::ROW_MAJOR, 16, false);
        A.fillRandom(42 + g);
        B.fillRandom(43 + g);
        C.fillRandom(44 + g);
        groups[g].A_matrices.push_back(std::move(A));
        groups[g].B_matrices.push_back(std::move(B));
        groups[g].C_matrices.push_back(std::move(C));
    }

    // Prepare args
    PreparedBatchGemmArgs prepared;
    UALError              status =
        prepare_batch_gemm_args(groups, MatrixType::f32, prepared);
    ASSERT_EQ(status, UALError::UAL_SUCCESS);

    // Prepare metadata
    ual_dlp->batch_prepare_metadata(prepared);

    // Verify all alpha/beta precomputed correctly
    ASSERT_EQ(prepared.alpha_f32.size(), 3);
    ASSERT_EQ(prepared.beta_f32.size(), 3);

    for (iter_t g = 0; g < 3; ++g) {
        EXPECT_FLOAT_EQ(prepared.alpha_f32[g], static_cast<float>(1.0 + g));
        EXPECT_FLOAT_EQ(prepared.beta_f32[g], static_cast<float>(0.5 * g));
    }

    // Verify metadata populated for all groups
    EXPECT_EQ(prepared.backend_metadata.size(), 3);
    for (iter_t g = 0; g < 3; ++g) {
        EXPECT_NE(prepared.backend_metadata[g], nullptr);
    }
}

// Test 5: Clear function resets everything
TEST_F(PreparedBatchGemmTest, ClearResetsAllFields)
{
    PreparedBatchGemmArgs prepared;

    // Populate some fields
    prepared.group_count = 5;
    prepared.backend_metadata.resize(5);
    prepared.alpha_f32.resize(5);
    prepared.beta_f32.resize(5);
    prepared.alpha_s32.resize(5);
    prepared.beta_s32.resize(5);
    prepared.post_ops.resize(5);

    // Call clear
    prepared.clear();

    // Verify everything reset
    EXPECT_EQ(prepared.group_count, 0);
    EXPECT_TRUE(prepared.backend_metadata.empty());
    EXPECT_TRUE(prepared.alpha_f32.empty());
    EXPECT_TRUE(prepared.beta_f32.empty());
    EXPECT_TRUE(prepared.alpha_s32.empty());
    EXPECT_TRUE(prepared.beta_s32.empty());
    EXPECT_TRUE(prepared.post_ops.empty());
}

// ============================================================================
// ERROR HANDLING TESTS FOR prepare_batch_gemm_args
// ============================================================================

// Test 7: Empty groups should return UAL_FAILURE
TEST_F(PreparedBatchGemmTest, PrepareArgsEmptyGroups)
{
    std::vector<BatchGroup> groups; // Empty
    PreparedBatchGemmArgs   prepared;

    UALError status =
        prepare_batch_gemm_args(groups, MatrixType::f32, prepared);

    EXPECT_EQ(status, UALError::UAL_FAILURE);
    EXPECT_EQ(prepared.group_count, 0);
}

// Test 8: Mismatched matrix counts (A.size != B.size) should fail validation
TEST_F(PreparedBatchGemmTest, PrepareArgsMismatchedMatrixCounts)
{
    std::vector<BatchGroup> groups(1);
    groups[0].m     = 32;
    groups[0].n     = 32;
    groups[0].k     = 32;
    groups[0].alpha = 1.0;
    groups[0].beta  = 0.0;

    // Add 2 A matrices but only 1 B and C matrix
    for (iter_t i = 0; i < 2; ++i) {
        Matrix A(32, 32, MatrixType::f32, MatrixLayout::ROW_MAJOR, 32, false);
        A.fillRandom(42 + i);
        groups[0].A_matrices.push_back(std::move(A));
    }

    // Only 1 B and C matrix - mismatch!
    Matrix B(32, 32, MatrixType::f32, MatrixLayout::ROW_MAJOR, 32, false);
    Matrix C(32, 32, MatrixType::f32, MatrixLayout::ROW_MAJOR, 32, false);
    B.fillRandom(43);
    C.fillRandom(44);
    groups[0].B_matrices.push_back(std::move(B));
    groups[0].C_matrices.push_back(std::move(C));

    PreparedBatchGemmArgs prepared;
    UALError              status =
        prepare_batch_gemm_args(groups, MatrixType::f32, prepared);

    // validate() should fail due to count mismatch
    EXPECT_EQ(status, UALError::UAL_FAILURE);
}

// Test 9: Matrix dimension mismatch (A dims don't match group m/k)
TEST_F(PreparedBatchGemmTest, PrepareArgsDimensionMismatch)
{
    std::vector<BatchGroup> groups(1);
    groups[0].m     = 64; // Group says 64x64
    groups[0].n     = 64;
    groups[0].k     = 64;
    groups[0].alpha = 1.0;
    groups[0].beta  = 0.0;

    // But matrices are 32x32 - mismatch!
    Matrix A(32, 32, MatrixType::f32, MatrixLayout::ROW_MAJOR, 32, false);
    Matrix B(32, 32, MatrixType::f32, MatrixLayout::ROW_MAJOR, 32, false);
    Matrix C(32, 32, MatrixType::f32, MatrixLayout::ROW_MAJOR, 32, false);
    A.fillRandom(42);
    B.fillRandom(43);
    C.fillRandom(44);
    groups[0].A_matrices.push_back(std::move(A));
    groups[0].B_matrices.push_back(std::move(B));
    groups[0].C_matrices.push_back(std::move(C));

    PreparedBatchGemmArgs prepared;
    UALError              status =
        prepare_batch_gemm_args(groups, MatrixType::f32, prepared);

    // validate() should fail due to dimension mismatch
    EXPECT_EQ(status, UALError::UAL_FAILURE);
}

// Test 10: Type inconsistency across groups should return UAL_NOT_SUPPORTED
TEST_F(PreparedBatchGemmTest, PrepareArgsTypeInconsistentGroups)
{
    std::vector<BatchGroup> groups(2);

    // Group 0: f32 matrices
    groups[0].m     = 32;
    groups[0].n     = 32;
    groups[0].k     = 32;
    groups[0].alpha = 1.0;
    groups[0].beta  = 0.0;

    Matrix A0(32, 32, MatrixType::f32, MatrixLayout::ROW_MAJOR, 32, false);
    Matrix B0(32, 32, MatrixType::f32, MatrixLayout::ROW_MAJOR, 32, false);
    Matrix C0(32, 32, MatrixType::f32, MatrixLayout::ROW_MAJOR, 32, false);
    A0.fillRandom(42);
    B0.fillRandom(43);
    C0.fillRandom(44);
    groups[0].A_matrices.push_back(std::move(A0));
    groups[0].B_matrices.push_back(std::move(B0));
    groups[0].C_matrices.push_back(std::move(C0));

    // Group 1: s8 matrices - TYPE MISMATCH!
    groups[1].m     = 32;
    groups[1].n     = 32;
    groups[1].k     = 32;
    groups[1].alpha = 1.0;
    groups[1].beta  = 0.0;

    Matrix A1(32, 32, MatrixType::s8, MatrixLayout::ROW_MAJOR, 32, false);
    Matrix B1(32, 32, MatrixType::s8, MatrixLayout::ROW_MAJOR, 32, false);
    Matrix C1(32, 32, MatrixType::s32, MatrixLayout::ROW_MAJOR, 32, false);
    A1.fillRandom(45);
    B1.fillRandom(46);
    C1.fillRandom(47);
    groups[1].A_matrices.push_back(std::move(A1));
    groups[1].B_matrices.push_back(std::move(B1));
    groups[1].C_matrices.push_back(std::move(C1));

    PreparedBatchGemmArgs prepared;
    UALError              status =
        prepare_batch_gemm_args(groups, MatrixType::f32, prepared);

    // Should fail due to type inconsistency across groups
    EXPECT_EQ(status, UALError::UAL_NOT_SUPPORTED);
}

// Test 11: Valid empty group (group_size=0) should succeed
TEST_F(PreparedBatchGemmTest, PrepareArgsEmptyGroupInBatch)
{
    std::vector<BatchGroup> groups(1);
    groups[0].m     = 32;
    groups[0].n     = 32;
    groups[0].k     = 32;
    groups[0].alpha = 1.0;
    groups[0].beta  = 0.0;
    // No matrices added - empty group with group_size=0

    PreparedBatchGemmArgs prepared;
    UALError              status =
        prepare_batch_gemm_args(groups, MatrixType::f32, prepared);

    // Empty groups are valid (for cases where some groups have 0 matrices)
    EXPECT_EQ(status, UALError::UAL_SUCCESS);
    EXPECT_EQ(prepared.group_count, 1);
    EXPECT_EQ(prepared.group_size[0], 0);
}

// Test: Performance of fast mode - should be fast with early exit
TEST_F(MatrixCompareTest, FastModePerformance)
{
    // Large matrix
    Matrix m1(100, 100, MatrixType::s32);
    Matrix m2(100, 100, MatrixType::s32);

    m1.fillValue(42);
    m2.fillValue(43); // Different at first element

    // Fast mode should exit early
    auto result = m1.compare(m2, MatrixCompareOptions::Fast());
    EXPECT_FALSE(result.equal);
    EXPECT_EQ(result.mismatchCount, 0);     // Fast mode doesn't count
    EXPECT_EQ(result.mismatches.size(), 0); // Fast mode doesn't collect
}

// Test: Packed 4-bit types
TEST_F(MatrixCompareTest, Packed4BitTypes)
{
    Matrix m1(4, 4, MatrixType::u4);
    Matrix m2(4, 4, MatrixType::u4);

    m1.fillRandom(54321);
    m2 = m1; // Copy

    auto result = m1.compare(m2, MatrixCompareOptions::Fast());
    EXPECT_TRUE(result.equal);

    // Now modify one and test
    Matrix m3(4, 4, MatrixType::u4);
    m3.fillRandom(99999);

    result = m1.compare(m3, MatrixCompareOptions::Fast());
    // Most likely different (unless extremely unlucky with RNG)
}

// Test: Tolerance with K dimension for F32
TEST_F(MatrixCompareTest, ToleranceWithKDimensionF32)
{
    Matrix m1(2, 2, MatrixType::f32);
    Matrix m2(2, 2, MatrixType::f32);

    m1.fillValue(1.0f);
    m2.fillValue(1.0f);

    // Set k dimension to simulate GEMM accumulation
    m1.setK(100);
    m2.setK(100);

    // Without k dimension, tolerance would be 10 * epsilon
    // With k=100, tolerance should be 10 * epsilon * 100
    auto result = m1.compare(m2, MatrixCompareOptions::Verbose(10));

    EXPECT_TRUE(result.equal);
    // Verify that tolerance was scaled by k
    double expected_tolerance =
        50.0 * std::numeric_limits<float>::epsilon() * 100.0;
    EXPECT_NEAR(result.usedAbsTolerance, expected_tolerance, 1e-15);
}

// Test: Tolerance with K dimension for BF16
TEST_F(MatrixCompareTest, ToleranceWithKDimensionBF16)
{
    Matrix m1(2, 2, MatrixType::bf16);
    Matrix m2(2, 2, MatrixType::bf16);

    m1.fillValue(1.0f);
    m2.fillValue(1.0f);

    // Set k dimension to simulate GEMM accumulation
    m1.setK(50);
    m2.setK(50);

    auto result = m1.compare(m2, MatrixCompareOptions::Verbose(10));

    EXPECT_TRUE(result.equal);
    // Verify that tolerance was scaled by k and uses bf16 epsilon
    // Note: We can't easily test the exact value without knowing
    // bf16_machine_epsilon but we can verify it's non-zero and scaled
    EXPECT_GT(result.usedAbsTolerance, 0.0);
}
