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

#pragma once

#include "classic/dlp_base_types.h"
#include "framework/cartesian_product.hh"
#include "framework/iterator.hh"
#include "framework/operation.hh"
#include "framework/simple_product.hh"
#include "framework/types.hh"
#include "framework/utils/matrix_tag.hh"
#include "framework/utils/postops_iterator.hh"

#include <any>
#include <cstddef>
#include <memory>
#include <vector>

namespace dlp::testing::utils {

using namespace dlp::testing::framework;

/**
 * @struct FillValueConfig
 * @brief Configuration for matrix initialization from YAML fill_value field
 *
 * This structure holds the parameters for custom matrix initialization:
 * - lb: Lower bound for random values
 * - ub: Upper bound for random values
 * - dist: Distribution type ("uniform" or "normal")
 * - force_int_distribution: Force integer-only values for float/bf16 matrices
 */
struct FillValueConfig
{
    double      lb   = -5.0;      ///< Lower bound for random values
    double      ub   = 5.0;       ///< Upper bound for random values
    std::string dist = "uniform"; ///< Distribution type
    bool        force_int_distribution =
        true; ///< Force integer-only random values for float types

    /**
     * @brief Default constructor with default values
     */
    FillValueConfig() = default;

    /**
     * @brief Constructor with custom values
     */
    FillValueConfig(double             l,
                    double             u,
                    const std::string& d,
                    bool               force_int = true)
        : lb(l)
        , ub(u)
        , dist(d)
        , force_int_distribution(force_int)
    {
    }
};

/**
 * @enum PatternType
 * @brief Types of fill patterns supported
 */
enum class PatternType : uint8_t
{
    Static,  ///< Explicit list of values
    Modulo,  ///< Modulo pattern: (x % range) + offset
    Linear,  ///< Linear pattern: x * multiplier + offset
    Sequence ///< Simple sequence: x with step
};

/**
 * @struct FillPatternConfig
 * @brief Configuration for pattern-based matrix initialization
 */
struct FillPatternConfig
{
    PatternType type = PatternType::Static;

    // Static pattern
    std::vector<double> values;

    // Expression-based patterns
    double lb         = 0.0;  ///< Lower bound / start value
    double ub         = 10.0; ///< Upper bound / end value
    double step       = 1.0;  ///< Increment step
    double multiplier = 1.0;  ///< Multiplier for Linear type
    double offset     = 0.0;  ///< Offset for Modulo/Linear type

    static constexpr size_t MAX_PATTERN_SIZE = 10000;

    /**
     * @brief Default constructor with default values
     */
    FillPatternConfig() = default;

    /**
     * @brief Generate pattern values based on configuration
     * @return Vector of pattern values
     * @throws std::runtime_error if configuration is invalid
     */
    std::vector<double> generatePattern() const;

    /**
     * @brief Validate pattern configuration
     * @param error_msg Output parameter for error message
     * @return true if valid, false otherwise
     */
    bool validate(std::string& error_msg) const;
};

/**
 * @brief Output stream support for PatternType (for debugging)
 */
std::ostream&
operator<<(std::ostream& os, PatternType type);

/**
 * @brief Output stream support for FillPatternConfig (for debugging)
 */
std::ostream&
operator<<(std::ostream& os, const FillPatternConfig& cfg);

/**
 * @struct ToleranceConfig
 * @brief Configuration for tolerance multipliers used in matrix comparisons
 *
 * This structure holds the multipliers for relative and absolute tolerance
 * calculations. The tolerance formula is:
 * - Relative: rel_tolerance = epsilon(dtype) * k * relative
 * - Absolute: abs_tolerance = epsilon(dtype) * k * absolute
 * Setting a value to 0 enforces bitwise equality (no tolerance).
 * Negative values indicate that default behavior should be used.
 */
struct ToleranceConfig
{
    double relative =
        -1.0; ///< Relative tolerance multiplier (-1 = use default)
    double absolute =
        -1.0; ///< Absolute tolerance multiplier (-1 = use default)

    /**
     * @brief Default constructor with default values
     */
    ToleranceConfig() = default;

    /**
     * @brief Constructor with custom values
     */
    ToleranceConfig(double rel, double abs)
        : relative(rel)
        , absolute(abs)
    {
    }
};

/**
 * @enum YieldType
 * @brief Defines the strategy for generating test parameter combinations
 *
 * This enumeration specifies how multiple parameter values should be combined
 * to generate test cases. The choice affects the number and nature of test
 * combinations generated.
 */
enum class YieldType : uint8_t
{
    CARTESIAN_PRODUCT = 0, ///< Generate all possible combinations of parameter
                           ///< values (m×n×k×... combinations)
    SIMPLE_PRODUCT = 1,    ///< Generate simple product combinations (requires
                           ///< equal-length parameter lists)
};

/**
 * @struct TestCaseIterators
 * @brief Container for all parameter iterators used in GEMM test cases
 *
 * This structure holds type-erased iterators for all possible GEMM test
 * parameters. Each iterator can generate one or more values for its
 * corresponding parameter. The iterators are used by MicroTest to generate
 * parameter combinations.
 *
 * @note All iterators use TypeErasedIterator which supports proper copy
 * semantics, so this struct can be safely copied and assigned.
 */
struct TestCaseIterators
{
    TypeErasedIterator a_type; ///< Iterator for matrix A data type (MatrixType)
    TypeErasedIterator b_type; ///< Iterator for matrix B data type (MatrixType)
    TypeErasedIterator c_type; ///< Iterator for matrix C data type (MatrixType)
    TypeErasedIterator
        acc_type; ///< Iterator for accumulation data type (MatrixType)
    TypeErasedIterator
        storage_format;         ///< Iterator for memory layout (MatrixLayout)
    TypeErasedIterator m;       ///< Iterator for matrix dimension M (md_t)
    TypeErasedIterator n;       ///< Iterator for matrix dimension N (md_t)
    TypeErasedIterator k;       ///< Iterator for matrix dimension K (md_t)
    TypeErasedIterator lda;     ///< Iterator for leading dimension of A (md_t)
    TypeErasedIterator ldb;     ///< Iterator for leading dimension of B (md_t)
    TypeErasedIterator ldc;     ///< Iterator for leading dimension of C (md_t)
    TypeErasedIterator alpha;   ///< Iterator for scaling factor alpha (double)
    TypeErasedIterator beta;    ///< Iterator for scaling factor beta (double)
    TypeErasedIterator trans_a; ///< Iterator for transpose flag of A (bool)
    TypeErasedIterator trans_b; ///< Iterator for transpose flag of B (bool)
    TypeErasedIterator mtag_a;  ///< Iterator for matrix tag of A (MatrixTag)
    TypeErasedIterator mtag_b;  ///< Iterator for matrix tag of B (MatrixTag)

    // Optional batch GEMM configuration (only used for batch_gemm_tests)
    bool has_group_size = false; ///< Whether group_size is present
    TypeErasedIterator
        group_size; ///< Iterator for group sizes (md_t) - batch GEMM only

    // Optional fill_value configuration
    bool            has_fill_value = false; ///< Whether fill_value is present
    FillValueConfig fill_value; ///< Fill value configuration if present

    // Optional fill_pattern configuration
    bool has_fill_pattern = false;  ///< Whether fill_pattern is present
    FillPatternConfig fill_pattern; ///< Fill pattern configuration if present

    // Optional tolerance configuration
    bool            has_tolerances = false; ///< Whether tolerance is present
    ToleranceConfig tolerances; ///< Tolerance configuration if present

    /**
     * @brief Default constructor - creates default-constructed
     * TypeErasedIterators
     *
     * All iterators are initialized to empty state. They should be populated
     * with actual iterator implementations before use.
     */
    TestCaseIterators() = default;

    // Copy constructor and assignment operator are automatically generated
    // and work correctly due to TypeErasedIterator's proper copy semantics
};

/**
 * @class MicroTest
 * @brief Manages test parameter combinations and provides type-safe access to
 * current values
 *
 * The MicroTest class is responsible for generating and navigating through test
 * parameter combinations. It can operate in two modes:
 * - CARTESIAN_PRODUCT: Generates all possible combinations of parameter values
 * - SIMPLE_PRODUCT: Generates element-wise combinations (requires equal-length
 * parameter lists)
 *
 * The class provides type-safe getter methods for all GEMM parameters and
 * handles the complexity of managing multiple iterators and their combinations.
 *
 * @note The parameter mapping is fixed and follows a specific order (see
 * private member docs). Changing the order requires updating both the
 * constructor logic and getter methods.
 *
 * Example usage:
 * @code
 * TestCaseIterators iterators;
 * // ... populate iterators with parameter values
 *
 * MicroTest test(iterators, YieldType::CARTESIAN_PRODUCT);
 * do {
 *     auto m = test.getM();
 *     auto n = test.getN();
 *     auto alpha = test.getAlpha();
 *     // ... use parameter values for testing
 *
 *     test.next();
 * } while (test.hasNext());
 * @endcode
 */
class MicroTest
{
  public:
    /**
     * @brief Default constructor - creates an uninitialized MicroTest
     *
     * The MicroTest will be in an invalid state until properly initialized
     * through assignment or by calling one of the other constructors.
     */
    MicroTest() = default;

    /**
     * @brief Construct MicroTest from a vector of iterators
     *
     * @param iterators Vector of TypeErasedIterators in the expected order
     * @param is_cartesian_product Strategy for combining parameter values
     *
     * This constructor expects iterators to be provided in the exact order
     * defined by the parameter mapping (see private member documentation).
     * The MicroTest will initialize with the first combination available.
     */
    MicroTest(std::vector<TypeErasedIterator>& iterators,
              YieldType                        is_cartesian_product)
        : m_iterators(iterators)
        , m_is_cartesian_product(is_cartesian_product)
        , m_index(0)
    {
        if (is_cartesian_product == YieldType::CARTESIAN_PRODUCT) {
            m_p_cartesian_product =
                std::make_unique<CartesianProduct>(m_iterators);
            // Initialize with the first combination
            if (m_p_cartesian_product->has_next()) {
                m_current_mt = m_p_cartesian_product->next();
            }
        } else {
            m_p_simple_product = std::make_unique<SimpleProduct>(m_iterators);
            // Initialize with the first combination
            if (m_p_simple_product->has_next()) {
                m_current_mt = m_p_simple_product->next();
            }
        }
    }
    MicroTest(TestCaseIterators& test_case_iterators,
              YieldType          is_cartesian_product)
        : m_is_cartesian_product(is_cartesian_product)
        , m_index(0)
    {
        m_test_case_iterators = test_case_iterators;

        // Build the iterators vector (18 total: 17 base GEMM params + 1
        // group_size for batch)
        m_iterators.reserve(18);
        m_iterators.push_back(m_test_case_iterators.a_type);
        m_iterators.push_back(m_test_case_iterators.b_type);
        m_iterators.push_back(m_test_case_iterators.c_type);
        m_iterators.push_back(m_test_case_iterators.acc_type);
        m_iterators.push_back(m_test_case_iterators.storage_format);
        m_iterators.push_back(m_test_case_iterators.m);
        m_iterators.push_back(m_test_case_iterators.n);
        m_iterators.push_back(m_test_case_iterators.k);
        m_iterators.push_back(m_test_case_iterators.lda);
        m_iterators.push_back(m_test_case_iterators.ldb);
        m_iterators.push_back(m_test_case_iterators.ldc);
        m_iterators.push_back(m_test_case_iterators.alpha);
        m_iterators.push_back(m_test_case_iterators.beta);
        m_iterators.push_back(m_test_case_iterators.trans_a);
        m_iterators.push_back(m_test_case_iterators.trans_b);
        m_iterators.push_back(m_test_case_iterators.mtag_a);
        m_iterators.push_back(m_test_case_iterators.mtag_b);
        m_iterators.push_back(m_test_case_iterators.group_size); // Index 17

        // Initialize the appropriate product
        if (is_cartesian_product == YieldType::CARTESIAN_PRODUCT) {
            m_p_cartesian_product =
                std::make_unique<CartesianProduct>(m_iterators);
            // Initialize with the first combination
            if (m_p_cartesian_product->has_next()) {
                m_current_mt = m_p_cartesian_product->next();
            }
        } else {
            m_p_simple_product = std::make_unique<SimpleProduct>(m_iterators);
            // Initialize with the first combination
            if (m_p_simple_product->has_next()) {
                m_current_mt = m_p_simple_product->next();
            }
        }
    }

    /**
     * @brief Construct MicroTest with PostOps support
     *
     * @param test_case_iterators Container with all parameter iterators
     * @param is_cartesian_product Strategy for combining parameter values
     * @param postops_iterator Iterator for PostOps combinations (can be
     * nullptr)
     */
    MicroTest(TestCaseIterators&               test_case_iterators,
              YieldType                        is_cartesian_product,
              std::unique_ptr<PostOpsIterator> postops_iterator)
        : m_is_cartesian_product(is_cartesian_product)
        , m_index(0)
        , m_postops_iterator(std::move(postops_iterator))
        , m_has_postops(m_postops_iterator != nullptr)
    {
        m_test_case_iterators = test_case_iterators;

        // Build the iterators vector (18 total: 17 base GEMM params + 1
        // group_size for batch)
        m_iterators.reserve(18);
        m_iterators.push_back(m_test_case_iterators.a_type);
        m_iterators.push_back(m_test_case_iterators.b_type);
        m_iterators.push_back(m_test_case_iterators.c_type);
        m_iterators.push_back(m_test_case_iterators.acc_type);
        m_iterators.push_back(m_test_case_iterators.storage_format);
        m_iterators.push_back(m_test_case_iterators.m);
        m_iterators.push_back(m_test_case_iterators.n);
        m_iterators.push_back(m_test_case_iterators.k);
        m_iterators.push_back(m_test_case_iterators.lda);
        m_iterators.push_back(m_test_case_iterators.ldb);
        m_iterators.push_back(m_test_case_iterators.ldc);
        m_iterators.push_back(m_test_case_iterators.alpha);
        m_iterators.push_back(m_test_case_iterators.beta);
        m_iterators.push_back(m_test_case_iterators.trans_a);
        m_iterators.push_back(m_test_case_iterators.trans_b);
        m_iterators.push_back(m_test_case_iterators.mtag_a);
        m_iterators.push_back(m_test_case_iterators.mtag_b);
        m_iterators.push_back(m_test_case_iterators.group_size); // Index 17

        // Initialize the appropriate product
        if (is_cartesian_product == YieldType::CARTESIAN_PRODUCT) {
            m_p_cartesian_product =
                std::make_unique<CartesianProduct>(m_iterators);
            // Initialize with the first combination
            if (m_p_cartesian_product->has_next()) {
                m_current_mt = m_p_cartesian_product->next();
            }
        } else {
            m_p_simple_product = std::make_unique<SimpleProduct>(m_iterators);
            // Initialize with the first combination
            if (m_p_simple_product->has_next()) {
                m_current_mt = m_p_simple_product->next();
            }

            // Validate PostOps count for SIMPLE_PRODUCT mode
            if (m_has_postops) {
                validateSimpleProductPostOps();
            }
        }
    }

    // Parameter accessor methods - all const and type-safe

    /**
     * @brief Get the data type for matrix A
     * @return MatrixType The data type (f32, bf16, s8, etc.)
     * @throws std::runtime_error if type casting fails
     */
    MatrixType getAType() const
    {
        try {
            return std::any_cast<MatrixType>(m_current_mt[0]);
        } catch (const std::bad_any_cast& e) {
            throw std::runtime_error("Failed to cast A type: "
                                     + std::string(e.what()));
        }
    }

    /**
     * @brief Get the data type for matrix B
     * @return MatrixType The data type (f32, bf16, s8, etc.)
     * @throws std::runtime_error if type casting fails
     */
    MatrixType getBType() const
    {
        try {
            return std::any_cast<MatrixType>(m_current_mt[1]);
        } catch (const std::bad_any_cast& e) {
            throw std::runtime_error("Failed to cast B type: "
                                     + std::string(e.what()));
        }
    }

    /**
     * @brief Get the data type for matrix C
     * @return MatrixType The data type (f32, bf16, s8, etc.)
     * @throws std::runtime_error if type casting fails
     */
    MatrixType getCType() const
    {
        try {
            return std::any_cast<MatrixType>(m_current_mt[2]);
        } catch (const std::bad_any_cast& e) {
            throw std::runtime_error("Failed to cast C type: "
                                     + std::string(e.what()));
        }
    }

    /**
     * @brief Get the accumulation data type
     * @return MatrixType The data type used for internal accumulation
     * @throws std::runtime_error if type casting fails
     */
    MatrixType getAccType() const
    {
        try {
            return std::any_cast<MatrixType>(m_current_mt[3]);
        } catch (const std::bad_any_cast& e) {
            throw std::runtime_error("Failed to cast Acc type: "
                                     + std::string(e.what()));
        }
    }

    /**
     * @brief Get the storage format (memory layout)
     * @return MatrixLayout ROW_MAJOR or COLUMN_MAJOR
     * @throws std::runtime_error if type casting fails
     */
    MatrixLayout getStorageFormat() const
    {
        try {
            return std::any_cast<MatrixLayout>(m_current_mt[4]);
        } catch (const std::bad_any_cast& e) {
            throw std::runtime_error("Failed to cast storage format: "
                                     + std::string(e.what()));
        }
    }

    /**
     * @brief Get the M dimension (number of rows in A and C)
     * @return md_t The M dimension value
     * @throws std::runtime_error if type casting fails
     */
    md_t getM() const
    {
        try {
            return std::any_cast<md_t>(m_current_mt[5]);
        } catch (const std::bad_any_cast& e) {
            throw std::runtime_error("Failed to cast M: "
                                     + std::string(e.what()));
        }
    }

    /**
     * @brief Get the N dimension (number of columns in B and C)
     * @return md_t The N dimension value
     * @throws std::runtime_error if type casting fails
     */
    md_t getN() const
    {
        try {
            return std::any_cast<md_t>(m_current_mt[6]);
        } catch (const std::bad_any_cast& e) {
            throw std::runtime_error("Failed to cast N: "
                                     + std::string(e.what()));
        }
    }

    /**
     * @brief Get the K dimension (inner dimension for multiplication)
     * @return md_t The K dimension value
     * @throws std::runtime_error if type casting fails
     */
    md_t getK() const
    {
        try {
            return std::any_cast<md_t>(m_current_mt[7]);
        } catch (const std::bad_any_cast& e) {
            throw std::runtime_error("Failed to cast K: "
                                     + std::string(e.what()));
        }
    }

    /**
     * @brief Get the leading dimension of matrix A
     * @return md_t The LDA value (stride between rows in row-major layout)
     * @throws std::runtime_error if type casting fails
     */
    md_t getLDA() const
    {
        try {
            return std::any_cast<md_t>(m_current_mt[8]);
        } catch (const std::bad_any_cast& e) {
            throw std::runtime_error("Failed to cast LDA: "
                                     + std::string(e.what()));
        }
    }

    /**
     * @brief Get the leading dimension of matrix B
     * @return md_t The LDB value (stride between rows in row-major layout)
     * @throws std::runtime_error if type casting fails
     */
    md_t getLDB() const
    {
        try {
            return std::any_cast<md_t>(m_current_mt[9]);
        } catch (const std::bad_any_cast& e) {
            throw std::runtime_error("Failed to cast LDB: "
                                     + std::string(e.what()));
        }
    }

    /**
     * @brief Get the leading dimension of matrix C
     * @return md_t The LDC value (stride between rows in row-major layout)
     * @throws std::runtime_error if type casting fails
     */
    md_t getLDC() const
    {
        try {
            return std::any_cast<md_t>(m_current_mt[10]);
        } catch (const std::bad_any_cast& e) {
            throw std::runtime_error("Failed to cast LDC: "
                                     + std::string(e.what()));
        }
    }

    /**
     * @brief Get the alpha scaling factor
     * @return double The alpha value used in C = alpha*A*B + beta*C
     * @throws std::runtime_error if type casting fails
     */
    double getAlpha() const
    {
        try {
            return std::any_cast<double>(m_current_mt[11]);
        } catch (const std::bad_any_cast& e) {
            throw std::runtime_error("Failed to cast Alpha: "
                                     + std::string(e.what()));
        }
    }

    /**
     * @brief Get the beta scaling factor
     * @return double The beta value used in C = alpha*A*B + beta*C
     * @throws std::runtime_error if type casting fails
     */
    double getBeta() const
    {
        try {
            return std::any_cast<double>(m_current_mt[12]);
        } catch (const std::bad_any_cast& e) {
            throw std::runtime_error("Failed to cast Beta: "
                                     + std::string(e.what()));
        }
    }

    /**
     * @brief Get the transpose flag for matrix A
     * @return bool true if A should be transposed, false otherwise
     * @throws std::runtime_error if type casting fails
     */
    bool getTransA() const
    {
        try {
            return std::any_cast<bool>(m_current_mt[13]);
        } catch (const std::bad_any_cast& e) {
            throw std::runtime_error("Failed to cast TransA: "
                                     + std::string(e.what()));
        }
    }

    /**
     * @brief Get the transpose flag for matrix B
     * @return bool true if B should be transposed, false otherwise
     * @throws std::runtime_error if type casting fails
     */
    bool getTransB() const
    {
        try {
            return std::any_cast<bool>(m_current_mt[14]);
        } catch (const std::bad_any_cast& e) {
            throw std::runtime_error("Failed to cast TransB: "
                                     + std::string(e.what()));
        }
    }

    /**
     * @brief Get the reorder flag for matrix A
     * @return bool true if A should be reordered, false otherwise
     * @throws std::runtime_error if type casting fails
     */
    bool getReorderA() const
    {
        try {
            auto tag =
                std::any_cast<MatrixTag>(m_current_mt[15]); // mtag_a index
            return tag == MatrixTag::REORDER;
        } catch (const std::bad_any_cast& e) {
            throw std::runtime_error("Failed to cast ReorderA: "
                                     + std::string(e.what()));
        }
    }

    /**
     * @brief Get the reorder flag for matrix B
     * @return bool true if B should be reordered, false otherwise
     * @throws std::runtime_error if type casting fails
     */
    bool getReorderB() const
    {
        try {
            auto tag =
                std::any_cast<MatrixTag>(m_current_mt[16]); // mtag_b index
            return tag == MatrixTag::REORDER;
        } catch (const std::bad_any_cast& e) {
            throw std::runtime_error("Failed to cast ReorderB: "
                                     + std::string(e.what()));
        }
    }

    /**
     * @brief Get the packing flag for matrix A
     * @return bool true if A should be packed, false otherwise
     * @throws std::runtime_error if type casting fails
     */
    bool getPackA() const
    {
        try {
            auto tag = std::any_cast<MatrixTag>(
                m_current_mt[15]); // mtag_a index (same as reorderA)
            return tag == MatrixTag::PACK;
        } catch (const std::bad_any_cast& e) {
            throw std::runtime_error("Failed to cast PackA: "
                                     + std::string(e.what()));
        }
    }

    /**
     * @brief Get the packing flag for matrix B
     * @return bool true if B should be packed, false otherwise
     * @throws std::runtime_error if type casting fails
     */
    bool getPackB() const
    {
        try {
            auto tag = std::any_cast<MatrixTag>(
                m_current_mt[16]); // mtag_b index (same as reorderB)
            return tag == MatrixTag::PACK;
        } catch (const std::bad_any_cast& e) {
            throw std::runtime_error("Failed to cast PackB: "
                                     + std::string(e.what()));
        }
    }

    /**
     * @brief Check if group_size parameter is present (for batch GEMM tests)
     * @return bool True if group_size was specified in YAML, false otherwise
     */
    bool hasGroupSize() const { return m_test_case_iterators.has_group_size; }

    /**
     * @brief Get group_size for batch GEMM tests
     * @return md_t Group size value (number of matrices in group)
     * @note Returns 1 (safe default) if group_size is not present
     */
    md_t getGroupSize() const
    {
        if (!m_test_case_iterators.has_group_size) {
            return 1; // Safe default for regular GEMM tests
        }
        try {
            return std::any_cast<md_t>(m_current_mt[17]); // group_size index
        } catch (const std::bad_any_cast& e) {
            throw std::runtime_error("Failed to cast GroupSize: "
                                     + std::string(e.what()));
        }
    }

    /**
     * @brief Get the current iteration index
     * @return size_t The index of the current parameter combination (0-based)
     */
    size_t getIndex() const { return m_index; }

    /**
     * @brief Check if fill_value configuration is present
     * @return bool True if fill_value is configured, false otherwise
     */
    bool hasFillValue() const { return m_test_case_iterators.has_fill_value; }

    /**
     * @brief Get the fill_value configuration
     * @return FillValueConfig The fill value configuration
     */
    const FillValueConfig& getFillValue() const
    {
        return m_test_case_iterators.fill_value;
    }

    /**
     * @brief Check if fill_pattern configuration is present
     * @return bool True if fill_pattern is configured, false otherwise
     */
    bool hasFillPattern() const
    {
        return m_test_case_iterators.has_fill_pattern;
    }

    /**
     * @brief Get the fill_pattern configuration
     * @return FillPatternConfig The fill pattern configuration
     */
    const FillPatternConfig& getFillPattern() const
    {
        return m_test_case_iterators.fill_pattern;
    }

    /**
     * @brief Get PostOps operation for the specified UAL type
     * @param ual_type The UAL type (DLP or REF) to create operation for
     * @return Ready-to-use IOperation object, or nullptr if no PostOps
     */
    std::shared_ptr<IOperation> getPostOp(UALType ual_type) const;

    /**
     * @brief Check if tolerance configuration is present
     * @return bool True if tolerance is configured, false otherwise
     */
    bool hasTolerances() const { return m_test_case_iterators.has_tolerances; }

    /**
     * @brief Get the tolerance configuration
     * @return ToleranceConfig The tolerance configuration
     */
    const ToleranceConfig& getTolerances() const
    {
        return m_test_case_iterators.tolerances;
    }

    /**
     * @brief Get the yield type (product mode) for this MicroTest
     * @return YieldType The yield type (CARTESIAN_PRODUCT or SIMPLE_PRODUCT)
     */
    YieldType getYieldType() const { return m_is_cartesian_product; }

    // Navigation and control methods

    /**
     * @brief Reset the iterator to the first parameter combination
     *
     * This method resets all internal iterators to their initial state
     * and sets the current combination to the first available combination.
     * PostOps iterator is also reset if enabled.
     */
    void reset()
    {
        // Reset PostOps iterator if enabled
        if (m_has_postops) {
            m_postops_iterator->reset();
        }

        // Reset base parameter iterators
        if ((m_is_cartesian_product == YieldType::CARTESIAN_PRODUCT)
            && m_p_cartesian_product) {
            // TODO: Implement CartesianProduct::reset() method
            // m_p_cartesian_product->reset();
        } else if (m_is_cartesian_product == YieldType::SIMPLE_PRODUCT
                   && m_p_simple_product) {
            // TODO: Implement SimpleProduct::reset() method
            // m_p_simple_product->reset();
        }
        m_index = 0;
    }

    /**
     * @brief Advance to the next parameter combination
     *
     * Moves the MicroTest to the next available combination of parameter
     * values. The specific algorithm depends on the YieldType (cartesian
     * product vs element-wise). Also increments the internal index counter.
     *
     * CARTESIAN_PRODUCT mode:
     *   - With PostOps: Nested iteration (PostOps × GEMM params)
     *   - PostOps advance first (inner loop), then GEMM params advance
     *
     * SIMPLE_PRODUCT mode (multi-group):
     *   - PostOps advance WITH GEMM params (synchronized)
     *   - Both iterators move together to next group
     * CARTESIAN_PRODUCT mode:
     *   - With PostOps: Nested iteration (PostOps × GEMM params)
     *   - PostOps advance first (inner loop), then GEMM params advance
     *
     * SIMPLE_PRODUCT mode (multi-group):
     *   - PostOps advance WITH GEMM params (synchronized)
     *   - Both iterators move together to next group
     */
    void next()
    {
        // In CARTESIAN mode: iterate PostOps × GEMM (nested loops)
        if (m_is_cartesian_product == YieldType::CARTESIAN_PRODUCT) {
            // Priority 1: Advance PostOps if available
            if (m_has_postops && m_postops_iterator->hasNext()) {
                m_postops_iterator->next();
                m_index++;
                return;
            }

            // Priority 2: Reset PostOps and advance base parameters
            if (m_has_postops) {
                m_postops_iterator->reset();
            }
        }

        // Advance base parameter combination (GEMM params / groups)
        if (m_is_cartesian_product == YieldType::CARTESIAN_PRODUCT
            && m_p_cartesian_product) {
            m_current_mt = m_p_cartesian_product->next();
        } else if (m_is_cartesian_product == YieldType::SIMPLE_PRODUCT
                   && m_p_simple_product) {
            m_current_mt = m_p_simple_product->next();

            // In SIMPLE mode: PostOps advance WITH GEMM params (synchronized)
            if (m_has_postops && m_postops_iterator->hasNext()) {
                m_postops_iterator->next();
            }
        }
        m_index++;
    }

    /**
     * @brief Get the total number of parameter combinations
     *
     * @return size_t Total number of combinations that will be generated
     *
     * For cartesian products, this is the product of all individual iterator
     * sizes. For element-wise combinations, this is the size of the longest
     * iterator (all iterators should have the same size in element-wise mode).
     *
     * CARTESIAN_PRODUCT mode with PostOps:
     *   Total = base_combinations * postops_combinations
     *
     * SIMPLE_PRODUCT mode with PostOps:
     *   Total = min(base_combinations, postops_combinations)
     *   (they iterate in lockstep/synchronized)
     */
    size_t getSize() const
    {
        size_t base_size = 0;

        if (m_is_cartesian_product == YieldType::CARTESIAN_PRODUCT
            && m_p_cartesian_product) {
            base_size = m_p_cartesian_product->size();
        } else if (m_is_cartesian_product == YieldType::SIMPLE_PRODUCT
                   && m_p_simple_product) {
            base_size = m_p_simple_product->size();
        }

        // CARTESIAN: base_size × PostOps (full cartesian product)
        // SIMPLE: base_size (PostOps are per-group, synchronized with GEMM
        // params)
        if (m_has_postops) {
            size_t postops_size = m_postops_iterator->getSize();
            if (m_is_cartesian_product == YieldType::CARTESIAN_PRODUCT) {
                return base_size * postops_size;
            } else {
                // SIMPLE: return base_size (iterate all groups, PostOps
                // sync/reuse)
                return base_size;
            }
        }

        return base_size;
    }

    /**
     * @brief Check if more parameter combinations are available
     *
     * @return true if next() can be called to get more combinations, false
     * otherwise
     *
     * This method should be checked before calling next() to avoid advancing
     * past the end of available combinations.
     *
     * When PostOps are enabled, checks if we haven't reached the total
     * expected combinations (base × PostOps).
     */
    bool hasNext() const
    {
        // When PostOps are enabled, use size-based checking to avoid extra
        // iterations
        if (m_has_postops) {
            return m_index < (getSize() - 1);
        }

        // For regular cases, check if base parameters have more combinations
        if (m_is_cartesian_product == YieldType::CARTESIAN_PRODUCT
            && m_p_cartesian_product) {
            return m_p_cartesian_product->has_next();
        } else if (m_is_cartesian_product == YieldType::SIMPLE_PRODUCT
                   && m_p_simple_product) {
            return m_p_simple_product->has_next();
        }
        return false;
    }

  private:
    /**
     * @brief Current parameter combination values
     *
     * Vector containing the current values for all parameters, stored as
     * std::any for type erasure. The order matches the parameter mapping
     * defined below.
     */
    std::vector<std::any> m_current_mt;

    /**
     * @brief Collection of all parameter iterators
     *
     * Vector of TypeErasedIterators in the order defined by the parameter
     * mapping. Built from m_test_case_iterators in the appropriate constructor.
     */
    std::vector<TypeErasedIterator> m_iterators;

    /**
     * @brief Named storage for parameter iterators
     *
     * TestCaseIterators struct that provides named access to individual
     * parameter iterators. This is copied from the constructor parameter for
     * reference.
     */
    TestCaseIterators m_test_case_iterators;

    /**
     * @brief Cartesian product generator (when using CARTESIAN_PRODUCT mode)
     *
     * Manages the generation of all possible parameter combinations using
     * an odometer-style algorithm. Only valid when m_is_cartesian_product
     * is YieldType::CARTESIAN_PRODUCT.
     */
    std::unique_ptr<CartesianProduct> m_p_cartesian_product;

    /**
     * @brief Simple product generator (when using SIMPLE_PRODUCT mode)
     *
     * Manages element-wise parameter combination generation. Only valid when
     * m_is_cartesian_product is YieldType::SIMPLE_PRODUCT.
     */
    std::unique_ptr<SimpleProduct> m_p_simple_product;

    /**
     * @brief Strategy for generating parameter combinations
     *
     * Determines whether to use cartesian product (all combinations) or
     * element-wise (corresponding elements) combination generation.
     */
    YieldType m_is_cartesian_product;

    /**
     * @brief Current iteration index
     *
     * 0-based index of the current parameter combination. Incremented
     * by next() method calls.
     */
    size_t m_index;

    /**
     * @brief PostOps iterator for generating operation combinations
     *
     * Manages the generation of PostOps combinations. Can be nullptr if no
     * PostOps are configured for this test case.
     */
    std::unique_ptr<PostOpsIterator> m_postops_iterator;

    /**
     * @brief Flag indicating whether PostOps are enabled for this test case
     */
    bool m_has_postops = false;

    /**
     * @brief Validate PostOps count for SIMPLE_PRODUCT mode
     *
     * Ensures that in SIMPLE_PRODUCT mode, the number of PostOps combinations
     * matches the number of GEMM parameter combinations, or is exactly 1
     * (broadcast pattern). Throws an error if validation fails.
     *
     * @throws std::runtime_error if PostOps count is invalid for SIMPLE mode
     */
    void validateSimpleProductPostOps() const
    {
        if (!m_has_postops || !m_p_simple_product) {
            return;
        }

        size_t base_size    = m_p_simple_product->size();
        size_t postops_size = m_postops_iterator->getSize();

        // Allow 1-to-N broadcast: single PostOps configuration for all groups
        if (postops_size == 1) {
            return; // Broadcast pattern is valid
        }

        if (base_size == 1) {
            return; // Broadcast pattern is valid
        }

        // Require exact match otherwise
        if (base_size != postops_size) {
            throw std::runtime_error(
                "SIMPLE_PRODUCT mode validation failed:\n"
                "  GEMM parameter combinations: "
                + std::to_string(base_size)
                + "\n"
                  "  PostOps Parameter combinations: "
                + std::to_string(postops_size)
                + "\n"
                  "In SIMPLE_PRODUCT mode, PostOps Parameter count must "
                  "either:\n"
                  "  1. Equal 1 (broadcast same PostOps to all groups)\n"
                  "  2. Exactly match GEMM count (one PostOps per group)\n"
                  "Current configuration is invalid. Please update your YAML "
                  "configuration.");
        }
    }

    /*
     * Parameter Vector Position Mapping:
     *
     * The m_current_mt vector and m_iterators vector follow this fixed order:
     *
     * Index | Parameter     | Type         | Description
     * ------|---------------|--------------|----------------------------------
     * 0     | a_type        | MatrixType   | Matrix A data type
     * 1     | b_type        | MatrixType   | Matrix B data type
     * 2     | c_type        | MatrixType   | Matrix C data type
     * 3     | acc_type      | MatrixType   | Accumulation data type
     * 4     | storage_format| MatrixLayout | Memory layout (row/column major)
     * 5     | m             | md_t         | Matrix rows dimension
     * 6     | n             | md_t         | Matrix columns dimension
     * 7     | k             | md_t         | Inner dimension
     * 8     | lda           | md_t         | Leading dimension of A
     * 9     | ldb           | md_t         | Leading dimension of B
     * 10    | ldc           | md_t         | Leading dimension of C
     * 11    | alpha         | double       | Scaling factor alpha
     * 12    | beta          | double       | Scaling factor beta
     * 13    | trans_a       | bool         | Transpose flag for A
     * 14    | trans_b       | bool         | Transpose flag for B
     * 15    | mtag_a        | MatrixTag    | Matrix tag for A
     * (reorder/pack/none) 16    | mtag_b        | MatrixTag    | Matrix tag for
     * B (reorder/pack/none) 17    | group_size    | md_t         | Group size
     * for batch GEMM
     *
     * Note: getReorderA()/getPackA() both check index 15 for different
     * MatrixTag values getReorderB()/getPackB() both check index 16 for
     * different MatrixTag values
     *
     * WARNING: This mapping is used by all getter methods. If the order
     * changes, all getXXX() methods must be updated to use the new indices.
     */

    /**
     * @brief Create operation parameter from PostOp configuration
     * @param config PostOp configuration from YAML
     * @param param_indices Map from parameter name to array index for
     * extraction
     * @return Operation parameter ready to be added to IOperation
     */
    std::unique_ptr<IOperationParam> createOperationParam(
        const PostOpsIterator::PostOpConfig& config,
        const std::map<std::string, size_t>& param_indices) const;

    /**
     * @brief Convert string to MatrixType enum
     * @param str String representation of matrix type
     * @return MatrixType enum value
     */
    MatrixType stringToMatrixType(const std::string& str) const;

    /**
     * @brief Convert MatrixType enum to string
     * @param type MatrixType enum value
     * @return String representation of matrix type
     */
    std::string matrixTypeToString(MatrixType type) const;

    /**
     * @brief Helper function to extract double parameter from PostOp
     * configuration
     * @param param_name Name of the parameter to extract
     * @param params Map of parameters from PostOp configuration
     * @param default_value Default value to use if parameter is not found
     * @return Extracted double value or default_value
     */
    double extractDoubleParam(
        const std::string&                                  param_name,
        const std::map<std::string, std::vector<std::any>>& params,
        double default_value) const;

    /**
     * @brief Helper function to extract MatrixType parameter from PostOp
     * configuration
     * @param param_name Name of the parameter to extract
     * @param params Map of parameters from PostOp configuration
     * @param default_type Default type to use if parameter is not found
     * (defaults to f32)
     * @return Extracted MatrixType or default_type
     */
    MatrixType extractMatrixTypeParam(
        const std::string&                                  param_name,
        const std::map<std::string, std::vector<std::any>>& params,
        MatrixType default_type = MatrixType::f32) const;

    /**
     * @brief Helper function to extract boolean parameter from PostOp
     * configuration
     * @param param_name Name of the parameter to extract
     * @param params Map of parameters from PostOp configuration
     * @param default_value Default value to use if parameter is not found
     * @return Extracted boolean value or default_value
     */
    bool extractBoolParam(
        const std::string&                                  param_name,
        const std::map<std::string, std::vector<std::any>>& params,
        bool default_value) const;
};

#if 0
// Old IParser
class IParser
{
  public:
    virtual MatrixType   getAType()           = 0;
    virtual MatrixType   getBType()           = 0;
    virtual MatrixType   getCType()           = 0;
    virtual MatrixType   getAccType()         = 0;
    virtual MatrixLayout getStorageFormat()   = 0;
    virtual int          getM()               = 0;
    virtual int          getN()               = 0;
    virtual int          getK()               = 0;
    virtual int          getLDA()             = 0;
    virtual int          getLDB()             = 0;
    virtual int          getLDC()             = 0;
    virtual int          getAlpha()           = 0;
    virtual int          getBeta()            = 0;
    virtual bool         getTransA()          = 0;
    virtual bool         getTransB()          = 0;
    virtual bool         getReorderA()        = 0;
    virtual bool         getReorderB()        = 0;
    virtual bool         getPackA()           = 0;
    virtual bool         getPackB()           = 0;
    virtual void         next()               = 0;
    virtual void         reset()              = 0;
    virtual size_t       getTestCount() const = 0;
};
#else
/**
 * @brief IParser interface which yields MicroTest objects.
 *
 * This interface provides methods to get a micro test, iterate through the
 * micro tests, and reset the parser.
 *
 * The product can be one in YieldType::CARTESIAN_PRODUCT or
 * YieldType::ELEMENT_WISE.
 *
 * In Cartesian product, the parser will yield a micro test for each
 combination
 * of the input matrices.
 *
 * In Element-wise, the parser will yield a micro test for each
 currosponding
 * value.
 *
 * Example:
 * n  = {1, 2, 3}
 * m  = {4, 5, 6}
 * k  = {7, 8, 9}
 *
 * In Cartesian product, the parser will yield the following micro tests:
 * {1, 4, 7}, {1, 4, 8}, {1, 4, 9}, {1, 5, 7}, {1, 5, 8}, {1, 5, 9}, {1, 6,
 7},
 * {1, 6, 8}, {1, 6, 9}, {2, 4, 7}, {2, 4, 8}, {2, 4, 9}, {2, 5, 7}, {2, 5,
 8},
 * {2, 5, 9}, {2, 6, 7}, {2, 6, 8}, {2, 6, 9}, {3, 4, 7}, {3, 4, 8}, {3, 4,
 9},
 * {3, 5, 7}, {3, 5, 8}, {3, 5, 9}, {3, 6, 7}, {3, 6, 8}, {3, 6, 9}
 *
 * In Element-wise, the parser will yield the following micro tests:
 * {1, 4, 7}, {1, 4, 8}, {1, 4, 9}, {2, 5, 7}, {2, 5, 8}, {2, 5, 9}, {3, 6,
 7},
 * {3, 6, 8}, {3, 6, 9}

 * There are more values in consideration:
 * - Multiple values for a_type, b_type, c_type, acc_type.
 * - Multiple values for m, n, k, lda, ldb, ldc.
 * - Multiple values for alpha, beta.
 * - Multiple values for trans_a, trans_b.
 * - Multiple values for mtag_a, mtag_b (each providing reorder/pack/none
 choices).
 *
 *
 * In Elementwise operation all sets should be equal, or only one element
 should
 * be in the set. That specific element will be repeated to match the size
 of
 * others sets with multiple values.
 *
 */

class IParser
{
  public:
    /**
     * @brief Get the current MicroTest instance
     * @return MicroTest& Mutable reference to current test (allows iteration
     * via next())
     */
    virtual MicroTest& getMicroTest()                     = 0;
    virtual void       next()                             = 0;
    virtual void       reset()                            = 0;
    virtual size_t     getMicroTestCount() const          = 0;
    virtual void       setYieldType(YieldType yield_type) = 0;
    virtual YieldType  getYieldType() const               = 0;
};
#endif

} // namespace dlp::testing::utils
