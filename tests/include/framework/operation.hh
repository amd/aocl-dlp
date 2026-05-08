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
#include <memory>
#include <stdexcept>
#include <vector>

namespace dlp::testing::framework {

/**
 * @enum OperationType
 * @brief Types of post-operations supported
 */
enum class OperationType : uint8_t
{
    ElementWise = 0,
    Bias        = 1,
    MatAdd      = 2,
    MatMul      = 3,
    Scale       = 4,
    A_Quant     = 5,
    WOQ         = 6, // Weight-Only Quantization for bf16s4
    B_Quant     = 7, // B matrix quantization (b_pre_quant / b_post_quant)
    GroupScale  = 8, // Group-level symmetric quantization scale factors
};

/**
 * @brief Check if an operation type is a pre-GEMM (quantisation) operation
 */
inline bool
isPreOp(OperationType type)
{
    return type == OperationType::A_Quant || type == OperationType::B_Quant
           || type == OperationType::WOQ || type == OperationType::GroupScale;
}

/**
 * @brief Check if an operation type is a post-GEMM (fusion) operation
 */
inline bool
isPostOp(OperationType type)
{
    return type == OperationType::ElementWise || type == OperationType::Bias
           || type == OperationType::Scale || type == OperationType::MatAdd
           || type == OperationType::MatMul;
}

/**
 * @brief Check if an operation type is GroupScale
 */
inline bool
isGroupScaleParam(OperationType type)
{
    return type == OperationType::GroupScale;
}

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
    Sigmoid   = 7,
    Mish      = 8
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
 * @class ScaleParam
 * @brief Parameter class for scale operations
 */
class ScaleParam : public IOperationParam
{
  private:
    std::unique_ptr<Matrix> m_scaleFactor;
    std::unique_ptr<Matrix> m_zeroPoint;

  public:
    ScaleParam() = default;

    ScaleParam(const ScaleParam& other)
    {
        if (other.m_scaleFactor) {
            m_scaleFactor = std::make_unique<Matrix>(*other.m_scaleFactor);
        }
        if (other.m_zeroPoint) {
            m_zeroPoint = std::make_unique<Matrix>(*other.m_zeroPoint);
        }
    }

    OperationType getType() const override { return OperationType::Scale; }

    std::unique_ptr<IOperationParam> clone() const override
    {
        return std::make_unique<ScaleParam>(*this);
    }

    void setScaleFactor(const Matrix& scale)
    {
        m_scaleFactor = std::make_unique<Matrix>(scale);
    }

    void setZeroPoint(const Matrix& zp)
    {
        m_zeroPoint = std::make_unique<Matrix>(zp);
    }

    const Matrix* getScaleFactor() const { return m_scaleFactor.get(); }
    const Matrix* getZeroPoint() const { return m_zeroPoint.get(); }
    bool          hasScaleFactor() const { return m_scaleFactor != nullptr; }
    bool          hasZeroPoint() const { return m_zeroPoint != nullptr; }
};

/**
 * @class QuantParam
 * @brief Parameter class for quantization operations
 */
class AQuantParam : public IOperationParam
{
  private:
    std::unique_ptr<Matrix> m_a_pre_op_sf;
    std::unique_ptr<Matrix> m_a_post_op_sf;
    std::unique_ptr<Matrix> m_a_pre_op_zp;
    std::unique_ptr<Matrix> m_a_post_op_zp;

  public:
    AQuantParam() = default;

    AQuantParam(const AQuantParam& other)
    {
        if (other.m_a_pre_op_sf) {
            m_a_pre_op_sf = std::make_unique<Matrix>(*other.m_a_pre_op_sf);
        }
        if (other.m_a_post_op_sf) {
            m_a_post_op_sf = std::make_unique<Matrix>(*other.m_a_post_op_sf);
        }
        if (other.m_a_pre_op_zp) {
            m_a_pre_op_zp = std::make_unique<Matrix>(*other.m_a_pre_op_zp);
        }
        if (other.m_a_post_op_zp) {
            m_a_post_op_zp = std::make_unique<Matrix>(*other.m_a_post_op_zp);
        }
    }

    OperationType getType() const override { return OperationType::A_Quant; }

    std::unique_ptr<IOperationParam> clone() const override
    {
        return std::make_unique<AQuantParam>(*this);
    }

    void setA_PreOpScaleFactor(const Matrix& sf)
    {
        m_a_pre_op_sf = std::make_unique<Matrix>(sf);
    }
    void setA_PostOpScaleFactor(const Matrix& sf)
    {
        m_a_post_op_sf = std::make_unique<Matrix>(sf);
    }
    void setA_PreOpZeroPoint(const Matrix& zp)
    {
        m_a_pre_op_zp = std::make_unique<Matrix>(zp);
    }
    void setA_PostOpZeroPoint(const Matrix& zp)
    {
        m_a_post_op_zp = std::make_unique<Matrix>(zp);
    }
    const Matrix* getA_PreOpScaleFactor() const { return m_a_pre_op_sf.get(); }
    const Matrix* getA_PostOpScaleFactor() const
    {
        return m_a_post_op_sf.get();
    }
    const Matrix* getA_PreOpZeroPoint() const { return m_a_pre_op_zp.get(); }
    const Matrix* getA_PostOpZeroPoint() const { return m_a_post_op_zp.get(); }
    bool hasA_PreOpScaleFactor() const { return m_a_pre_op_sf != nullptr; }
    bool hasA_PostOpScaleFactor() const { return m_a_post_op_sf != nullptr; }
    bool hasA_PreOpZeroPoint() const { return m_a_pre_op_zp != nullptr; }
    bool hasA_PostOpZeroPoint() const { return m_a_post_op_zp != nullptr; }
};

/**
 * @class WOQParam
 * @brief Parameter class for Weight-Only Quantization (WOQ) pre-operations
 * Used for bf16s4 quantization with scale factors and zero points for B matrix
 */
class WOQParam : public IOperationParam
{
  private:
    std::unique_ptr<Matrix> m_b_scale_factor;
    std::unique_ptr<Matrix> m_b_zero_point;

  public:
    WOQParam() = default;

    WOQParam(const WOQParam& other)
    {
        if (other.m_b_scale_factor) {
            m_b_scale_factor =
                std::make_unique<Matrix>(*other.m_b_scale_factor);
        }
        if (other.m_b_zero_point) {
            m_b_zero_point = std::make_unique<Matrix>(*other.m_b_zero_point);
        }
    }

    OperationType getType() const override { return OperationType::WOQ; }

    std::unique_ptr<IOperationParam> clone() const override
    {
        return std::make_unique<WOQParam>(*this);
    }

    void setB_ScaleFactor(const Matrix& sf)
    {
        m_b_scale_factor = std::make_unique<Matrix>(sf);
    }

    void setB_ZeroPoint(const Matrix& zp)
    {
        m_b_zero_point = std::make_unique<Matrix>(zp);
    }

    const Matrix* getB_ScaleFactor() const { return m_b_scale_factor.get(); }
    const Matrix* getB_ZeroPoint() const { return m_b_zero_point.get(); }
    bool hasB_ScaleFactor() const { return m_b_scale_factor != nullptr; }
    bool hasB_ZeroPoint() const { return m_b_zero_point != nullptr; }
};

/**
 * @class BQuantParam
 * @brief Parameter class for B matrix quantization operations
 * Mirrors AQuantParam but for the B matrix (b_pre_quant / b_post_quant)
 */
class BQuantParam : public IOperationParam
{
  private:
    std::unique_ptr<Matrix> m_b_pre_op_sf;
    std::unique_ptr<Matrix> m_b_post_op_sf;
    std::unique_ptr<Matrix> m_b_pre_op_zp;
    std::unique_ptr<Matrix> m_b_post_op_zp;
    md_t                    m_group_size = 0;

  public:
    BQuantParam() = default;

    BQuantParam(const BQuantParam& other)
        : m_group_size(other.m_group_size)
    {
        if (other.m_b_pre_op_sf) {
            m_b_pre_op_sf = std::make_unique<Matrix>(*other.m_b_pre_op_sf);
        }
        if (other.m_b_post_op_sf) {
            m_b_post_op_sf = std::make_unique<Matrix>(*other.m_b_post_op_sf);
        }
        if (other.m_b_pre_op_zp) {
            m_b_pre_op_zp = std::make_unique<Matrix>(*other.m_b_pre_op_zp);
        }
        if (other.m_b_post_op_zp) {
            m_b_post_op_zp = std::make_unique<Matrix>(*other.m_b_post_op_zp);
        }
    }

    OperationType getType() const override { return OperationType::B_Quant; }

    std::unique_ptr<IOperationParam> clone() const override
    {
        return std::make_unique<BQuantParam>(*this);
    }

    void setB_PreOpScaleFactor(const Matrix& sf)
    {
        m_b_pre_op_sf = std::make_unique<Matrix>(sf);
    }
    void setB_PostOpScaleFactor(const Matrix& sf)
    {
        m_b_post_op_sf = std::make_unique<Matrix>(sf);
    }
    void setB_PreOpZeroPoint(const Matrix& zp)
    {
        m_b_pre_op_zp = std::make_unique<Matrix>(zp);
    }
    void setB_PostOpZeroPoint(const Matrix& zp)
    {
        m_b_post_op_zp = std::make_unique<Matrix>(zp);
    }
    void          setGroupSize(md_t groupSize) { m_group_size = groupSize; }
    const Matrix* getB_PreOpScaleFactor() const { return m_b_pre_op_sf.get(); }
    const Matrix* getB_PostOpScaleFactor() const
    {
        return m_b_post_op_sf.get();
    }
    const Matrix* getB_PreOpZeroPoint() const { return m_b_pre_op_zp.get(); }
    const Matrix* getB_PostOpZeroPoint() const { return m_b_post_op_zp.get(); }
    md_t          getGroupSize() const { return m_group_size; }
    bool hasB_PreOpScaleFactor() const { return m_b_pre_op_sf != nullptr; }
    bool hasB_PostOpScaleFactor() const { return m_b_post_op_sf != nullptr; }
    bool hasB_PreOpZeroPoint() const { return m_b_pre_op_zp != nullptr; }
    bool hasB_PostOpZeroPoint() const { return m_b_post_op_zp != nullptr; }
};

/**
 * @class GroupScaleParam
 * @brief Parameter class for group-level symmetric quantization scale factors
 * Used for s8s8 symmetric quantization with A and B scale factors
 */
class GroupScaleParam : public IOperationParam
{
  private:
    std::unique_ptr<Matrix> m_a_scale_factor;
    std::unique_ptr<Matrix> m_b_scale_factor;
    md_t                    m_group_size = 0; // 0 means full k dimension

  public:
    GroupScaleParam() = default;

    GroupScaleParam(const GroupScaleParam& other)
        : m_group_size(other.m_group_size)
    {
        if (other.m_a_scale_factor) {
            m_a_scale_factor =
                std::make_unique<Matrix>(*other.m_a_scale_factor);
        }
        if (other.m_b_scale_factor) {
            m_b_scale_factor =
                std::make_unique<Matrix>(*other.m_b_scale_factor);
        }
    }

    OperationType getType() const override { return OperationType::GroupScale; }

    std::unique_ptr<IOperationParam> clone() const override
    {
        return std::make_unique<GroupScaleParam>(*this);
    }

    void setAScaleFactor(const Matrix& sf)
    {
        m_a_scale_factor = std::make_unique<Matrix>(sf);
    }

    void setBScaleFactor(const Matrix& sf)
    {
        m_b_scale_factor = std::make_unique<Matrix>(sf);
    }

    void setGroupSize(md_t groupSize) { m_group_size = groupSize; }

    const Matrix* getAScaleFactor() const { return m_a_scale_factor.get(); }
    const Matrix* getBScaleFactor() const { return m_b_scale_factor.get(); }
    bool hasAScaleFactor() const { return m_a_scale_factor != nullptr; }
    bool hasBScaleFactor() const { return m_b_scale_factor != nullptr; }
    md_t getGroupSize() const { return m_group_size; }
};

/**
 * @class BiasParam
 * @brief Parameter class for bias operations
 */
class BiasParam : public IOperationParam
{
  private:
    Matrix                  m_bias;
    std::unique_ptr<Matrix> m_scaleFactor;
    std::unique_ptr<Matrix> m_zeroPoint;

  public:
    BiasParam(const Matrix& bias)
        : m_bias(bias)
    {
    }

    BiasParam(const BiasParam& other)
        : m_bias(other.m_bias)
    {
        if (other.m_scaleFactor) {
            m_scaleFactor = std::make_unique<Matrix>(*other.m_scaleFactor);
        }
        if (other.m_zeroPoint) {
            m_zeroPoint = std::make_unique<Matrix>(*other.m_zeroPoint);
        }
    }

    OperationType getType() const override { return OperationType::Bias; }

    std::unique_ptr<IOperationParam> clone() const override
    {
        return std::make_unique<BiasParam>(*this);
    }

    const Matrix& getBias() const { return m_bias; }

    void setScaleFactor(const Matrix& scale)
    {
        m_scaleFactor = std::make_unique<Matrix>(scale);
    }

    void setZeroPoint(const Matrix& zp)
    {
        m_zeroPoint = std::make_unique<Matrix>(zp);
    }

    const Matrix* getScaleFactor() const { return m_scaleFactor.get(); }
    const Matrix* getZeroPoint() const { return m_zeroPoint.get(); }
    bool          hasScaleFactor() const { return m_scaleFactor != nullptr; }
    bool          hasZeroPoint() const { return m_zeroPoint != nullptr; }
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
    static constexpr bool supports_alpha =
        true; // SWISH requires alpha parameter
    static constexpr bool supports_beta = false;
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

template<>
struct ElementWiseTraits<ElementWiseOperation::Mish>
{
    static constexpr bool supports_alpha = false;
    static constexpr bool supports_beta  = false;
};

// (No traits needed for Scale)

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
  private:
    std::unique_ptr<Matrix> m_alpha;

  public:
    SwishBuilder& setAlpha(const Matrix& alpha)
    {
        m_alpha = std::make_unique<Matrix>(alpha);
        return *this;
    }

    std::unique_ptr<IOperationParam> build()
    {
        auto param =
            std::make_unique<ElementWiseParam>(ElementWiseOperation::Swish);
        if (m_alpha) {
            param->setAlpha(*m_alpha);
        } else {
            // Provide default alpha value for SWISH if not specified
            auto default_alpha = Matrix::fromValue(1.0f, MatrixType::f32);
            param->setAlpha(default_alpha);
        }
        return param;
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
 * @class MishBuilder
 * @brief Type-safe builder for Mish operations: x * tanh(softplus(x))
 */
class MishBuilder
{
  public:
    std::unique_ptr<IOperationParam> build()
    {
        return std::make_unique<ElementWiseParam>(ElementWiseOperation::Mish);
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
    std::unique_ptr<Matrix> m_zeroPoint;

  public:
    ScaleBuilder& setScaleFactor(const Matrix& scale)
    {
        m_scaleFactor = std::make_unique<Matrix>(scale);
        return *this;
    }

    ScaleBuilder& setZeroPoint(const Matrix& zp)
    {
        m_zeroPoint = std::make_unique<Matrix>(zp);
        return *this;
    }

    std::unique_ptr<IOperationParam> build()
    {
        if (!m_scaleFactor) {
            throw std::runtime_error(
                "Scale factor is required for Scale operation");
        }
        auto param = std::make_unique<ScaleParam>();
        param->setScaleFactor(*m_scaleFactor);
        if (m_zeroPoint) {
            param->setZeroPoint(*m_zeroPoint);
        }
        return param;
    }
};

/**
 * @class QuantBuilder
 * @brief Type-safe builder for Quant operations
 */
class AQuantBuilder
{
  private:
    std::unique_ptr<Matrix> m_a_pre_op_sf;
    std::unique_ptr<Matrix> m_a_post_op_sf;
    std::unique_ptr<Matrix> m_a_pre_op_zp;
    std::unique_ptr<Matrix> m_a_post_op_zp;

  public:
    AQuantBuilder& setA_PreOpScaleFactor(const Matrix& sf)
    {
        m_a_pre_op_sf = std::make_unique<Matrix>(sf);
        return *this;
    }
    AQuantBuilder& setA_PostOpScaleFactor(const Matrix& sf)
    {
        m_a_post_op_sf = std::make_unique<Matrix>(sf);
        return *this;
    }
    AQuantBuilder& setA_PreOpZeroPoint(const Matrix& zp)
    {
        m_a_pre_op_zp = std::make_unique<Matrix>(zp);
        return *this;
    }
    AQuantBuilder& setA_PostOpZeroPoint(const Matrix& zp)
    {
        m_a_post_op_zp = std::make_unique<Matrix>(zp);
        return *this;
    }
    std::unique_ptr<IOperationParam> build()
    {
        if (!m_a_pre_op_sf) {
            throw std::runtime_error(
                "A_PreOpScaleFactor is required for Quant operation");
        }
        if (!m_a_post_op_sf) {
            throw std::runtime_error(
                "A_PostOpScaleFactor is required for Quant operation");
        }
        auto param = std::make_unique<AQuantParam>();
        param->setA_PreOpScaleFactor(*m_a_pre_op_sf);
        param->setA_PostOpScaleFactor(*m_a_post_op_sf);
        if (m_a_pre_op_zp) {
            param->setA_PreOpZeroPoint(*m_a_pre_op_zp);
        }
        if (m_a_post_op_zp) {
            param->setA_PostOpZeroPoint(*m_a_post_op_zp);
        }
        return param;
    }
};

/**
 * @class BQuantBuilder
 * @brief Type-safe builder for B matrix quantization operations
 */
class BQuantBuilder
{
  private:
    std::unique_ptr<Matrix> m_b_pre_op_sf;
    std::unique_ptr<Matrix> m_b_post_op_sf;
    std::unique_ptr<Matrix> m_b_pre_op_zp;
    std::unique_ptr<Matrix> m_b_post_op_zp;
    md_t                    m_group_size = 0;

  public:
    BQuantBuilder& setB_PreOpScaleFactor(const Matrix& sf)
    {
        m_b_pre_op_sf = std::make_unique<Matrix>(sf);
        return *this;
    }
    BQuantBuilder& setB_PostOpScaleFactor(const Matrix& sf)
    {
        m_b_post_op_sf = std::make_unique<Matrix>(sf);
        return *this;
    }
    BQuantBuilder& setB_PreOpZeroPoint(const Matrix& zp)
    {
        m_b_pre_op_zp = std::make_unique<Matrix>(zp);
        return *this;
    }
    BQuantBuilder& setB_PostOpZeroPoint(const Matrix& zp)
    {
        m_b_post_op_zp = std::make_unique<Matrix>(zp);
        return *this;
    }
    BQuantBuilder& setGroupSize(md_t groupSize)
    {
        m_group_size = groupSize;
        return *this;
    }
    std::unique_ptr<IOperationParam> build()
    {
        if (!m_b_pre_op_sf) {
            throw std::runtime_error(
                "B_PreOpScaleFactor is required for B_Quant operation");
        }
        if (!m_b_post_op_sf) {
            throw std::runtime_error(
                "B_PostOpScaleFactor is required for B_Quant operation");
        }
        auto param = std::make_unique<BQuantParam>();
        param->setB_PreOpScaleFactor(*m_b_pre_op_sf);
        param->setB_PostOpScaleFactor(*m_b_post_op_sf);
        if (m_b_pre_op_zp) {
            param->setB_PreOpZeroPoint(*m_b_pre_op_zp);
        }
        if (m_b_post_op_zp) {
            param->setB_PostOpZeroPoint(*m_b_post_op_zp);
        }
        if (m_group_size > 0) {
            param->setGroupSize(m_group_size);
        }
        return param;
    }
};

/**
 * @class WOQBuilder
 * @brief Type-safe builder for Weight-Only Quantization (WOQ) pre-operations
 */
class WOQBuilder
{
  private:
    std::unique_ptr<Matrix> m_b_scale_factor;
    std::unique_ptr<Matrix> m_b_zero_point;

  public:
    WOQBuilder& setB_ScaleFactor(const Matrix& sf)
    {
        m_b_scale_factor = std::make_unique<Matrix>(sf);
        return *this;
    }

    WOQBuilder& setB_ZeroPoint(const Matrix& zp)
    {
        m_b_zero_point = std::make_unique<Matrix>(zp);
        return *this;
    }

    std::unique_ptr<IOperationParam> build()
    {
        if (!m_b_scale_factor) {
            throw std::runtime_error(
                "B_ScaleFactor is required for WOQ operation");
        }
        auto param = std::make_unique<WOQParam>();
        param->setB_ScaleFactor(*m_b_scale_factor);
        if (m_b_zero_point) {
            param->setB_ZeroPoint(*m_b_zero_point);
        }
        return param;
    }
};

/**
 * @class GroupScaleBuilder
 * @brief Type-safe builder for group-level symmetric quantization scale factors
 */
class GroupScaleBuilder
{
  private:
    std::unique_ptr<Matrix> m_a_scale_factor;
    std::unique_ptr<Matrix> m_b_scale_factor;
    md_t                    m_group_size = 0;

  public:
    GroupScaleBuilder& setAScaleFactor(const Matrix& sf)
    {
        m_a_scale_factor = std::make_unique<Matrix>(sf);
        return *this;
    }

    GroupScaleBuilder& setBScaleFactor(const Matrix& sf)
    {
        m_b_scale_factor = std::make_unique<Matrix>(sf);
        return *this;
    }

    GroupScaleBuilder& setGroupSize(md_t groupSize)
    {
        m_group_size = groupSize;
        return *this;
    }

    std::unique_ptr<IOperationParam> build()
    {
        if (!m_a_scale_factor || !m_b_scale_factor) {
            throw std::runtime_error(
                "GroupScaleBuilder: both A and B scale factors are required");
        }
        auto param = std::make_unique<GroupScaleParam>();
        param->setAScaleFactor(*m_a_scale_factor);
        param->setBScaleFactor(*m_b_scale_factor);
        if (m_group_size > 0) {
            param->setGroupSize(m_group_size);
        }
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
    std::unique_ptr<Matrix> m_scaleFactor;
    std::unique_ptr<Matrix> m_zeroPoint;

  public:
    BiasBuilder& setBias(const Matrix& bias)
    {
        m_bias = std::make_unique<Matrix>(bias);
        return *this;
    }

    BiasBuilder& setScaleFactor(const Matrix& scale)
    {
        m_scaleFactor = std::make_unique<Matrix>(scale);
        return *this;
    }

    BiasBuilder& setZeroPoint(const Matrix& zp)
    {
        m_zeroPoint = std::make_unique<Matrix>(zp);
        return *this;
    }

    std::unique_ptr<IOperationParam> build()
    {
        if (!m_bias) {
            throw std::runtime_error("Bias is required for Bias operation");
        }
        auto param = std::make_unique<BiasParam>(*m_bias);
        if (m_scaleFactor) {
            param->setScaleFactor(*m_scaleFactor);
        }
        if (m_zeroPoint) {
            param->setZeroPoint(*m_zeroPoint);
        }
        return param;
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
    inline MishBuilder createMish()
    {
        return MishBuilder{};
    }
    // Scale operation factories
    inline ScaleBuilder createScale()
    {
        return ScaleBuilder{};
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
    inline AQuantBuilder createAQuant()
    {
        return AQuantBuilder{};
    }
    inline BQuantBuilder createBQuant()
    {
        return BQuantBuilder{};
    }
    inline WOQBuilder createWOQ()
    {
        return WOQBuilder{};
    }
    inline GroupScaleBuilder createGroupScale()
    {
        return GroupScaleBuilder{};
    }

} // namespace postops

} // namespace dlp::testing::framework
