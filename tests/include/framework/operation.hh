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

#include "framework/matrix.hh"
#include "framework/ual.hh"
#include <memory>
#include <stdexcept>
#include <vector>

namespace dlp::testing::framework {

// Forward declarations
class IOperation;

/**
 * @enum OperationType
 * @brief Types of post-operations supported
 */
enum class OperationType : uint8_t
{
    ElementWise = 0,
    Sum         = 1,
    Bias        = 2,
    MatAdd      = 3,
    MatMul      = 4,
    Scale       = 5,
};

/**
 * @enum ElementWiseOperation
 * @brief Types of element-wise operations
 */
enum class ElementWiseOperation : uint8_t
{
    Relu      = 0,
    Prelu     = 1,
    Gelu_Tanh = 2,
    Gelu_Erf  = 3,
    Clip      = 4,
    Swish     = 5,
    Tanh      = 6,
    Sigmoid   = 7
};

/**
 * @enum SumOperation
 * @brief Types of sum/scale operations
 */
enum class SumOperation : uint8_t
{
    Sum   = 0,
    Scale = 1
};

/**
 * @class OperationParam
 * @brief Base class for all operation parameters
 */
class IOperationParam
{
  public:
    virtual ~IOperationParam()                               = default;
    virtual OperationType                    getType() const = 0;
    virtual std::unique_ptr<IOperationParam> clone() const   = 0;
};

/**
 * @class ElementWiseParam
 * @brief Parameter class for element-wise operations
 */
class ElementWiseParam : public IOperationParam
{
  private:
    ElementWiseOperation    m_operation;
    std::unique_ptr<Matrix> m_alpha;
    std::unique_ptr<Matrix> m_beta;

  public:
    ElementWiseParam(ElementWiseOperation op)
        : m_operation(op)
    {
    }

    ElementWiseParam(const ElementWiseParam& other)
        : m_operation(other.m_operation)
    {
        if (other.m_alpha) {
            m_alpha = std::make_unique<Matrix>(*other.m_alpha);
        }
        if (other.m_beta) {
            m_beta = std::make_unique<Matrix>(*other.m_beta);
        }
    }

    OperationType getType() const override
    {
        return OperationType::ElementWise;
    }

    std::unique_ptr<IOperationParam> clone() const override
    {
        return std::make_unique<ElementWiseParam>(*this);
    }

    ElementWiseOperation getOperation() const { return m_operation; }

    void setAlpha(const Matrix& alpha)
    {
        m_alpha = std::make_unique<Matrix>(alpha);
    }

    void setBeta(const Matrix& beta)
    {
        m_beta = std::make_unique<Matrix>(beta);
    }

    const Matrix* getAlpha() const { return m_alpha.get(); }
    const Matrix* getBeta() const { return m_beta.get(); }
    bool          hasAlpha() const { return m_alpha != nullptr; }
    bool          hasBeta() const { return m_beta != nullptr; }
};

/**
 * @class SumParam
 * @brief Parameter class for sum/scale operations
 */
class SumParam : public IOperationParam
{
  private:
    SumOperation            m_operation;
    std::unique_ptr<Matrix> m_scaleFactor;
    std::unique_ptr<Matrix> m_zeroPoint;
    bool                    m_isPowerOf2 = false;

  public:
    SumParam(SumOperation op)
        : m_operation(op)
    {
    }

    SumParam(const SumParam& other)
        : m_operation(other.m_operation)
        , m_isPowerOf2(other.m_isPowerOf2)
    {
        if (other.m_scaleFactor) {
            m_scaleFactor = std::make_unique<Matrix>(*other.m_scaleFactor);
        }
        if (other.m_zeroPoint) {
            m_zeroPoint = std::make_unique<Matrix>(*other.m_zeroPoint);
        }
    }

    OperationType getType() const override { return OperationType::Sum; }

    std::unique_ptr<IOperationParam> clone() const override
    {
        return std::make_unique<SumParam>(*this);
    }

    SumOperation getOperation() const { return m_operation; }

    void setScaleFactor(const Matrix& scale)
    {
        m_scaleFactor = std::make_unique<Matrix>(scale);
    }

    void setZeroPoint(const Matrix& zp)
    {
        m_zeroPoint = std::make_unique<Matrix>(zp);
    }

    void setIsPowerOf2(bool isPow2) { m_isPowerOf2 = isPow2; }

    const Matrix* getScaleFactor() const { return m_scaleFactor.get(); }
    const Matrix* getZeroPoint() const { return m_zeroPoint.get(); }
    bool          hasScaleFactor() const { return m_scaleFactor != nullptr; }
    bool          hasZeroPoint() const { return m_zeroPoint != nullptr; }
    bool          getIsPowerOf2() const { return m_isPowerOf2; }
};

/**
 * @class BiasParam
 * @brief Parameter class for bias operations
 */
class BiasParam : public IOperationParam
{
  private:
    Matrix m_bias;

  public:
    BiasParam(const Matrix& bias)
        : m_bias(bias)
    {
    }

    OperationType getType() const override { return OperationType::Bias; }

    std::unique_ptr<IOperationParam> clone() const override
    {
        return std::make_unique<BiasParam>(*this);
    }

    const Matrix& getBias() const { return m_bias; }
};

/**
 * @class MatrixAddParam
 * @brief Parameter class for matrix addition operations
 */
class MatrixAddParam : public IOperationParam
{
  private:
    Matrix                  m_matrix;
    std::unique_ptr<Matrix> m_scaleFactor;

  public:
    MatrixAddParam(const Matrix& matrix)
        : m_matrix(matrix)
    {
    }

    MatrixAddParam(const MatrixAddParam& other)
        : m_matrix(other.m_matrix)
    {
        if (other.m_scaleFactor) {
            m_scaleFactor = std::make_unique<Matrix>(*other.m_scaleFactor);
        }
    }

    OperationType getType() const override { return OperationType::MatAdd; }

    std::unique_ptr<IOperationParam> clone() const override
    {
        return std::make_unique<MatrixAddParam>(*this);
    }

    void setScaleFactor(const Matrix& scale)
    {
        m_scaleFactor = std::make_unique<Matrix>(scale);
    }

    const Matrix& getMatrix() const { return m_matrix; }
    const Matrix* getScaleFactor() const { return m_scaleFactor.get(); }
    bool          hasScaleFactor() const { return m_scaleFactor != nullptr; }
};

/**
 * @class MatrixMulParam
 * @brief Parameter class for matrix multiplication operations
 */
class MatrixMulParam : public IOperationParam
{
  private:
    Matrix                  m_matrix;
    std::unique_ptr<Matrix> m_scaleFactor;

  public:
    MatrixMulParam(const Matrix& matrix)
        : m_matrix(matrix)
    {
    }

    MatrixMulParam(const MatrixMulParam& other)
        : m_matrix(other.m_matrix)
    {
        if (other.m_scaleFactor) {
            m_scaleFactor = std::make_unique<Matrix>(*other.m_scaleFactor);
        }
    }

    OperationType getType() const override { return OperationType::MatMul; }

    std::unique_ptr<IOperationParam> clone() const override
    {
        return std::make_unique<MatrixMulParam>(*this);
    }

    void setScaleFactor(const Matrix& scale)
    {
        m_scaleFactor = std::make_unique<Matrix>(scale);
    }

    const Matrix& getMatrix() const { return m_matrix; }
    const Matrix* getScaleFactor() const { return m_scaleFactor.get(); }
    bool          hasScaleFactor() const { return m_scaleFactor != nullptr; }
};

// OPERATION PARAMETER TRAITS
/**
 * @brief Traits for element-wise operations to define supported parameters
 */
template<ElementWiseOperation Op>
struct ElementWiseTraits;

template<>
struct ElementWiseTraits<ElementWiseOperation::Relu>
{
    static constexpr bool supports_alpha = false;
    static constexpr bool supports_beta  = false;
};

template<>
struct ElementWiseTraits<ElementWiseOperation::Prelu>
{
    static constexpr bool supports_alpha = true;
    static constexpr bool supports_beta  = false;
};

template<>
struct ElementWiseTraits<ElementWiseOperation::Gelu_Tanh>
{
    static constexpr bool supports_alpha = false;
    static constexpr bool supports_beta  = false;
};

template<>
struct ElementWiseTraits<ElementWiseOperation::Gelu_Erf>
{
    static constexpr bool supports_alpha = false;
    static constexpr bool supports_beta  = false;
};

template<>
struct ElementWiseTraits<ElementWiseOperation::Clip>
{
    static constexpr bool supports_alpha = true; // lower bound
    static constexpr bool supports_beta  = true; // upper bound
};

template<>
struct ElementWiseTraits<ElementWiseOperation::Swish>
{
    static constexpr bool supports_alpha = false;
    static constexpr bool supports_beta  = false;
};

template<>
struct ElementWiseTraits<ElementWiseOperation::Tanh>
{
    static constexpr bool supports_alpha = false;
    static constexpr bool supports_beta  = false;
};

template<>
struct ElementWiseTraits<ElementWiseOperation::Sigmoid>
{
    static constexpr bool supports_alpha = false;
    static constexpr bool supports_beta  = false;
};

/**
 * @brief Traits for sum operations to define supported parameters
 */
template<SumOperation Op>
struct SumTraits;

template<>
struct SumTraits<SumOperation::Scale>
{
    static constexpr bool requires_scale_factor = true;
    static constexpr bool supports_zero_point   = false;
    static constexpr bool supports_power_of_2   = true;
};

template<>
struct SumTraits<SumOperation::Sum>
{
    static constexpr bool requires_scale_factor = false;
    static constexpr bool supports_zero_point   = true;
    static constexpr bool supports_power_of_2   = true;
};

// TYPE-SAFE BUILDER CLASSES
/**
 * @class ReluBuilder
 * @brief Type-safe builder for ReLU operations
 */
class ReluBuilder
{
  public:
    std::unique_ptr<IOperationParam> build()
    {
        return std::make_unique<ElementWiseParam>(ElementWiseOperation::Relu);
    }
};

/**
 * @class PreluBuilder
 * @brief Type-safe builder for PReLU operations
 */
class PreluBuilder
{
  private:
    std::unique_ptr<Matrix> m_alpha;

  public:
    PreluBuilder& setAlpha(const Matrix& alpha)
    {
        m_alpha = std::make_unique<Matrix>(alpha);
        return *this;
    }

    std::unique_ptr<IOperationParam> build()
    {
        if (!m_alpha) {
            throw std::runtime_error("Alpha parameter is required for PReLU");
        }
        auto param =
            std::make_unique<ElementWiseParam>(ElementWiseOperation::Prelu);
        param->setAlpha(*m_alpha);
        return param;
    }
};

/**
 * @class ClipBuilder
 * @brief Type-safe builder for Clip operations
 */
class ClipBuilder
{
  private:
    std::unique_ptr<Matrix> m_lowerBound;
    std::unique_ptr<Matrix> m_upperBound;

  public:
    ClipBuilder& setLowerBound(const Matrix& lower)
    {
        m_lowerBound = std::make_unique<Matrix>(lower);
        return *this;
    }

    ClipBuilder& setUpperBound(const Matrix& upper)
    {
        m_upperBound = std::make_unique<Matrix>(upper);
        return *this;
    }

    // Alias methods for clarity
    ClipBuilder& setAlpha(const Matrix& alpha) { return setLowerBound(alpha); }
    ClipBuilder& setBeta(const Matrix& beta) { return setUpperBound(beta); }

    std::unique_ptr<IOperationParam> build()
    {
        auto param =
            std::make_unique<ElementWiseParam>(ElementWiseOperation::Clip);
        if (m_lowerBound) {
            param->setAlpha(*m_lowerBound);
        }
        if (m_upperBound) {
            param->setBeta(*m_upperBound);
        }
        return param;
    }
};

/**
 * @class GeluTanhBuilder
 * @brief Type-safe builder for GeLU-Tanh operations
 */
class GeluTanhBuilder
{
  public:
    std::unique_ptr<IOperationParam> build()
    {
        return std::make_unique<ElementWiseParam>(
            ElementWiseOperation::Gelu_Tanh);
    }
};

/**
 * @class GeluErfBuilder
 * @brief Type-safe builder for GeLU-Erf operations
 */
class GeluErfBuilder
{
  public:
    std::unique_ptr<IOperationParam> build()
    {
        return std::make_unique<ElementWiseParam>(
            ElementWiseOperation::Gelu_Erf);
    }
};

/**
 * @class SwishBuilder
 * @brief Type-safe builder for Swish operations
 */
class SwishBuilder
{
  public:
    std::unique_ptr<IOperationParam> build()
    {
        return std::make_unique<ElementWiseParam>(ElementWiseOperation::Swish);
    }
};

/**
 * @class TanhBuilder
 * @brief Type-safe builder for Tanh operations
 */
class TanhBuilder
{
  public:
    std::unique_ptr<IOperationParam> build()
    {
        return std::make_unique<ElementWiseParam>(ElementWiseOperation::Tanh);
    }
};

/**
 * @class SigmoidBuilder
 * @brief Type-safe builder for Sigmoid operations
 */
class SigmoidBuilder
{
  public:
    std::unique_ptr<IOperationParam> build()
    {
        return std::make_unique<ElementWiseParam>(
            ElementWiseOperation::Sigmoid);
    }
};

/**
 * @class ScaleBuilder
 * @brief Type-safe builder for Scale operations
 */
class ScaleBuilder
{
  private:
    std::unique_ptr<Matrix> m_scaleFactor;
    bool                    m_isPowerOf2 = false;

  public:
    ScaleBuilder& setScaleFactor(const Matrix& scale)
    {
        m_scaleFactor = std::make_unique<Matrix>(scale);
        return *this;
    }

    ScaleBuilder& setIsPowerOf2(bool isPow2)
    {
        m_isPowerOf2 = isPow2;
        return *this;
    }

    std::unique_ptr<IOperationParam> build()
    {
        if (!m_scaleFactor) {
            throw std::runtime_error(
                "Scale factor is required for Scale operation");
        }
        auto param = std::make_unique<SumParam>(SumOperation::Scale);
        param->setScaleFactor(*m_scaleFactor);
        param->setIsPowerOf2(m_isPowerOf2);
        return param;
    }
};

/**
 * @class SumBuilder
 * @brief Type-safe builder for Sum operations
 */
class SumBuilder
{
  private:
    std::unique_ptr<Matrix> m_scaleFactor;
    std::unique_ptr<Matrix> m_zeroPoint;
    bool                    m_isPowerOf2 = false;

  public:
    SumBuilder& setScaleFactor(const Matrix& scale)
    {
        m_scaleFactor = std::make_unique<Matrix>(scale);
        return *this;
    }

    SumBuilder& setZeroPoint(const Matrix& zp)
    {
        m_zeroPoint = std::make_unique<Matrix>(zp);
        return *this;
    }

    SumBuilder& setIsPowerOf2(bool isPow2)
    {
        m_isPowerOf2 = isPow2;
        return *this;
    }

    std::unique_ptr<IOperationParam> build()
    {
        auto param = std::make_unique<SumParam>(SumOperation::Sum);
        if (m_scaleFactor) {
            param->setScaleFactor(*m_scaleFactor);
        }
        if (m_zeroPoint) {
            param->setZeroPoint(*m_zeroPoint);
        }
        param->setIsPowerOf2(m_isPowerOf2);
        return param;
    }
};

/**
 * @class BiasBuilder
 * @brief Type-safe builder for Bias operations
 */
class BiasBuilder
{
  private:
    std::unique_ptr<Matrix> m_bias;

  public:
    BiasBuilder& setBias(const Matrix& bias)
    {
        m_bias = std::make_unique<Matrix>(bias);
        return *this;
    }

    std::unique_ptr<IOperationParam> build()
    {
        if (!m_bias) {
            throw std::runtime_error("Bias is required for Bias operation");
        }
        return std::make_unique<BiasParam>(*m_bias);
    }
};

/**
 * @class MatrixAddBuilder
 * @brief Type-safe builder for Matrix Addition operations
 */
class MatrixAddBuilder
{
  private:
    std::unique_ptr<Matrix> m_matrix;
    std::unique_ptr<Matrix> m_scaleFactor;

  public:
    MatrixAddBuilder& setMatrix(const Matrix& matrix)
    {
        m_matrix = std::make_unique<Matrix>(matrix);
        return *this;
    }

    MatrixAddBuilder& setScaleFactor(const Matrix& scale)
    {
        m_scaleFactor = std::make_unique<Matrix>(scale);
        return *this;
    }

    std::unique_ptr<IOperationParam> build()
    {
        if (!m_matrix) {
            throw std::runtime_error(
                "Matrix is required for Matrix Add operation");
        }
        auto param = std::make_unique<MatrixAddParam>(*m_matrix);
        if (m_scaleFactor) {
            param->setScaleFactor(*m_scaleFactor);
        }
        return param;
    }
};

/**
 * @class MatrixMulBuilder
 * @brief Type-safe builder for Matrix Multiplication operations
 */
class MatrixMulBuilder
{
  private:
    std::unique_ptr<Matrix> m_matrix;
    std::unique_ptr<Matrix> m_scaleFactor;

  public:
    MatrixMulBuilder& setMatrix(const Matrix& matrix)
    {
        m_matrix = std::make_unique<Matrix>(matrix);
        return *this;
    }

    MatrixMulBuilder& setScaleFactor(const Matrix& scale)
    {
        m_scaleFactor = std::make_unique<Matrix>(scale);
        return *this;
    }

    std::unique_ptr<IOperationParam> build()
    {
        if (!m_matrix) {
            throw std::runtime_error(
                "Matrix is required for Matrix Mul operation");
        }
        auto param = std::make_unique<MatrixMulParam>(*m_matrix);
        if (m_scaleFactor) {
            param->setScaleFactor(*m_scaleFactor);
        }
        return param;
    }
};

// FACTORY FUNCTIONS FOR TYPE-SAFE CREATION
namespace postops {

    // Element-wise operation factories
    inline ReluBuilder createRelu()
    {
        return ReluBuilder{};
    }
    inline PreluBuilder createPrelu()
    {
        return PreluBuilder{};
    }
    inline ClipBuilder createClip()
    {
        return ClipBuilder{};
    }
    inline GeluTanhBuilder createGeluTanh()
    {
        return GeluTanhBuilder{};
    }
    inline GeluErfBuilder createGeluErf()
    {
        return GeluErfBuilder{};
    }
    inline SwishBuilder createSwish()
    {
        return SwishBuilder{};
    }
    inline TanhBuilder createTanh()
    {
        return TanhBuilder{};
    }
    inline SigmoidBuilder createSigmoid()
    {
        return SigmoidBuilder{};
    }

    // Sum/Scale operation factories
    inline ScaleBuilder createScale()
    {
        return ScaleBuilder{};
    }
    inline SumBuilder createSum()
    {
        return SumBuilder{};
    }

    // Other operation factories
    inline BiasBuilder createBias()
    {
        return BiasBuilder{};
    }
    inline MatrixAddBuilder createMatrixAdd()
    {
        return MatrixAddBuilder{};
    }
    inline MatrixMulBuilder createMatrixMul()
    {
        return MatrixMulBuilder{};
    }

} // namespace postops

// OPERATION PARAMETERS CONTAINER
/**
 * @class OperationParams
 * @brief Container for multiple operation parameters
 */
class OperationParams
{
  private:
    std::vector<std::unique_ptr<IOperationParam>> m_params;

  public:
    void add(std::unique_ptr<IOperationParam> param)
    {
        if (param) {
            m_params.push_back(std::move(param));
        }
    }

    void add(const IOperationParam& param)
    {
        m_params.push_back(param.clone());
    }

    void clear() { m_params.clear(); }

    size_t size() const { return m_params.size(); }

    bool empty() const { return m_params.empty(); }

    // Iterator support for range-based loops
    auto begin() const { return m_params.begin(); }
    auto end() const { return m_params.end(); }

    // Access by index
    const IOperationParam& operator[](size_t index) const
    {
        return *m_params[index];
    }

    const std::unique_ptr<IOperationParam>& getParam(size_t index) const
    {
        return m_params[index];
    }
};

// IOPERATION INTERFACE
/**
 * @class IOperation
 * @brief Interface for post-operation handling
 */
class IOperation
{
  protected:
    UALType                                       m_ual_type = UALType::REF;
    std::vector<std::unique_ptr<IOperationParam>> m_operation_params;

  public:
    virtual ~IOperation() = default;
    virtual UALType getUALType() const { return m_ual_type; }

    // Intuitive interface
    virtual void addOperations(const OperationParams& params)         = 0;
    virtual void addOperation(std::unique_ptr<IOperationParam> param) = 0;
    virtual void finalize()                                           = 0;

    // Access to parameters for backend implementations
    const std::vector<std::unique_ptr<IOperationParam>>& getParams() const
    {
        return m_operation_params;
    }
};

// OPERATION FACTORY
/**
 * @class OperationFactory
 * @brief Factory for creating UAL-specific operation objects
 */
class OperationFactory
{
  public:
    /**
     * @brief Create an operation object for the specified UAL type
     * @param type UAL type to create operation for
     * @return Shared pointer to the created operation
     */
    static std::shared_ptr<IOperation> createOperation(UALType type);
};

// Note: RefOperation is now implemented in framework/operation_ref.hh
// in the dlp::testing::classic namespace

} // namespace dlp::testing::framework
