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
#include "framework/cartesian_product.hh"
#include "framework/range.hh"
#include "framework/simple_product.hh"
#include "framework/value_iterable.hh"
#include "framework/vector_iterable.hh"

// Standard Headers
#include <gtest/gtest.h>
#include <vector>

using namespace dlp::testing;

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
    for (int i = 0; i < 8; i++) {
        cp.next(); // Skip to get to (1, 10.0f) then (2, 1.1f)
    }

    auto result11 = cp.next(); // (2, 1.1f)
    EXPECT_EQ(std::any_cast<int>(result11[0]), 2);
    EXPECT_FLOAT_EQ(std::any_cast<float>(result11[1]), 1.1f);

    // Skip to near the end
    for (int i = 0; i < 88; i++) {
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
    for (int i = 1; i <= 10; i++) {
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
    for (int i = 1; i <= 5; i++) {
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
#include "framework/matrix.hh"
#include "framework/operation.hh"
#include "framework/ual_ref.hh"

using namespace dlp::testing;
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
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            data[i * 6 + j] =
                i * 10.0f
                + j; // Row 0: 0,1,2,3  Row 1: 10,11,12,13  Row 2: 20,21,22,23
        }
    }

    // Reorder using reference implementation
    UalRef ual_ref;
    Matrix output;
    bool   result = ual_ref.reorder(input, output, MatrixType::f32);

    EXPECT_TRUE(result);

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
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            data[i * 5 + j] = i * 10.0f + j; // Physical: Row 0: 0,1,2,3  Row 1:
                                             // 10,11,12,13  Row 2: 20,21,22,23
        }
    }

    // Reorder using reference implementation
    UalRef ual_ref;
    Matrix output;
    bool   result = ual_ref.reorder(input, output, MatrixType::f32);

    EXPECT_TRUE(result);

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
    UalRef ual_ref;
    Matrix output;
    bool   result = ual_ref.reorder(input, output, MatrixType::f32);

    EXPECT_TRUE(result);

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

// Test sum/scale operation builders
TEST_F(PostOpsTest, SumScaleBuilders)
{
    // Test Scale (requires scale factor)
    auto scale_factor =
        Matrix::fromVector(std::vector<float>{ 1.0f, 2.0f, 0.5f });
    auto scale = postops::createScale()
                     .setScaleFactor(scale_factor)
                     .setIsPowerOf2(true)
                     .build();

    EXPECT_EQ(scale->getType(), OperationType::Sum);
    auto& scale_param = static_cast<const SumParam&>(*scale);
    EXPECT_EQ(scale_param.getOperation(), SumOperation::Scale);
    EXPECT_TRUE(scale_param.hasScaleFactor());
    EXPECT_FALSE(scale_param.hasZeroPoint());
    EXPECT_TRUE(scale_param.getIsPowerOf2());

    // Test Sum (optional parameters)
    auto sum_scale  = Matrix::fromValue(1.5f);
    auto zero_point = Matrix::fromValue(0);
    auto sum        = postops::createSum()
                   .setScaleFactor(sum_scale)
                   .setZeroPoint(zero_point)
                   .setIsPowerOf2(false)
                   .build();

    auto& sum_param = static_cast<const SumParam&>(*sum);
    EXPECT_EQ(sum_param.getOperation(), SumOperation::Sum);
    EXPECT_TRUE(sum_param.hasScaleFactor());
    EXPECT_TRUE(sum_param.hasZeroPoint());
    EXPECT_FALSE(sum_param.getIsPowerOf2());
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

// Test OperationParams container
TEST_F(PostOpsTest, OperationParamsContainer)
{
    OperationParams params;
    EXPECT_TRUE(params.empty());
    EXPECT_EQ(params.size(), 0);

    // Add operations
    auto relu         = postops::createRelu().build();
    auto alpha        = Matrix::fromValue(0.2f);
    auto prelu        = postops::createPrelu().setAlpha(alpha).build();
    auto scale_factor = Matrix::fromValue(2.0f);
    auto scale = postops::createScale().setScaleFactor(scale_factor).build();

    params.add(std::move(relu));
    params.add(std::move(prelu));
    params.add(std::move(scale));

    EXPECT_FALSE(params.empty());
    EXPECT_EQ(params.size(), 3);

    // Test access by index
    EXPECT_EQ(params[0].getType(), OperationType::ElementWise);
    EXPECT_EQ(params[1].getType(), OperationType::ElementWise);
    EXPECT_EQ(params[2].getType(), OperationType::Sum);

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

// Test operation factory
TEST_F(PostOpsTest, OperationFactory)
{
    // Test DLP operation creation
    auto dlp_operation = OperationFactory::createOperation(UALType::DLP);
    EXPECT_NE(dlp_operation, nullptr);
    EXPECT_EQ(dlp_operation->getUALType(), UALType::DLP);

    // Test REF operation creation
    auto ref_operation = OperationFactory::createOperation(UALType::REF);
    EXPECT_NE(ref_operation, nullptr);
    EXPECT_EQ(ref_operation->getUALType(), UALType::REF);

    // Test unsupported UAL type
    EXPECT_THROW(OperationFactory::createOperation(static_cast<UALType>(999)),
                 std::runtime_error);
}

// Test RefOperation functionality
TEST_F(PostOpsTest, RefOperationFunctionality)
{
    auto operation = OperationFactory::createOperation(UALType::REF);

    // Test adding individual operations
    auto relu  = postops::createRelu().build();
    auto alpha = Matrix::fromValue(0.1f);
    auto prelu = postops::createPrelu().setAlpha(alpha).build();

    operation->addOperation(std::move(relu));
    operation->addOperation(std::move(prelu));

    // Test finalization (should not throw)
    EXPECT_NO_THROW(operation->finalize());

    // Test adding operations after finalization (RefOperation allows this)
    auto scale_factor = Matrix::fromValue(2.0f);
    auto scale = postops::createScale().setScaleFactor(scale_factor).build();
    EXPECT_NO_THROW(operation->addOperation(std::move(scale)));
}

// Test DLP operation functionality
TEST_F(PostOpsTest, DlpOperationFunctionality)
{
    auto operation = OperationFactory::createOperation(UALType::DLP);

    // Test adding operations via OperationParams
    OperationParams params;

    auto relu         = postops::createRelu().build();
    auto alpha        = Matrix::fromValue(0.2f);
    auto prelu        = postops::createPrelu().setAlpha(alpha).build();
    auto scale_factor = Matrix::fromVector(std::vector<float>{ 1.0f, 2.0f });
    auto scale        = postops::createScale()
                     .setScaleFactor(scale_factor)
                     .setIsPowerOf2(true)
                     .build();

    params.add(std::move(relu));
    params.add(std::move(prelu));
    params.add(std::move(scale));

    operation->addOperations(params);

    // Test finalization
    EXPECT_NO_THROW(operation->finalize());

    // Test adding operations after finalization should throw
    auto bias_vec = Matrix::fromValue(0.5f);
    auto bias     = postops::createBias().setBias(bias_vec).build();
    EXPECT_THROW(operation->addOperation(std::move(bias)), std::runtime_error);
}

// Test complex operation sequence
TEST_F(PostOpsTest, ComplexOperationSequence)
{
    auto operation = OperationFactory::createOperation(UALType::DLP);

    // Create a complex sequence: Bias -> Scale -> PReLU -> Clip
    auto bias_vec = Matrix::fromVector(std::vector<float>{ 0.1f, 0.2f, 0.3f });
    auto bias     = postops::createBias().setBias(bias_vec).build();

    auto scale_factor =
        Matrix::fromVector(std::vector<float>{ 2.0f, 1.5f, 0.8f });
    auto scale = postops::createScale()
                     .setScaleFactor(scale_factor)
                     .setIsPowerOf2(false)
                     .build();

    auto alpha = Matrix::fromValue(0.01f);
    auto prelu = postops::createPrelu().setAlpha(alpha).build();

    auto lower = Matrix::fromValue(-6.0f);
    auto upper = Matrix::fromValue(6.0f);
    auto clip =
        postops::createClip().setLowerBound(lower).setUpperBound(upper).build();

    // Add operations in sequence
    operation->addOperation(std::move(bias));
    operation->addOperation(std::move(scale));
    operation->addOperation(std::move(prelu));
    operation->addOperation(std::move(clip));

    // Finalize
    EXPECT_NO_THROW(operation->finalize());

    // Verify UAL type
    EXPECT_EQ(operation->getUALType(), UALType::DLP);
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

// Test your vision: exactly as described in the task
TEST_F(PostOpsTest, YourVisionRealized)
{
    // Exactly as you envisioned!
    auto ew = postops::createPrelu()
                  .setAlpha(Matrix::fromValue(0.2f, MatrixType::f32))
                  .build();

    auto sum = postops::createScale()
                   .setScaleFactor(Matrix::fromVector(
                       std::vector<float>{ 1.0f, 2.0f }, MatrixType::f32))
                   .setIsPowerOf2(true)
                   .build();

    OperationParams params;
    params.add(std::move(ew));
    params.add(std::move(sum));

    // Backend integration
    auto operation = OperationFactory::createOperation(UALType::DLP);
    operation->addOperations(params);
    operation->finalize();

    // Verify the operation was created successfully
    EXPECT_EQ(operation->getUALType(), UALType::DLP);
    EXPECT_EQ(operation->getParams().size(), 2);

    // Verify first operation (PReLU)
    const auto& first_param = *operation->getParams()[0];
    EXPECT_EQ(first_param.getType(), OperationType::ElementWise);
    const auto& ew_param = static_cast<const ElementWiseParam&>(first_param);
    EXPECT_EQ(ew_param.getOperation(), ElementWiseOperation::Prelu);
    EXPECT_TRUE(ew_param.hasAlpha());

    // Verify second operation (Scale)
    const auto& second_param = *operation->getParams()[1];
    EXPECT_EQ(second_param.getType(), OperationType::Sum);
    const auto& sum_param = static_cast<const SumParam&>(second_param);
    EXPECT_EQ(sum_param.getOperation(), SumOperation::Scale);
    EXPECT_TRUE(sum_param.hasScaleFactor());
    EXPECT_TRUE(sum_param.getIsPowerOf2());
}
