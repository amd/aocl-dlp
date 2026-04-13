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

#include "x86_kernel_ops_generator.hh"

namespace amdzen::x86gen {

// ─────────────────────────────────────────────────────────────────────────
// kernelopsBase: CRTP base that enforces generateImpl() in derived strategies
// ─────────────────────────────────────────────────────────────────────────

template<typename Derived, utils::kernelInstrType KType>
class kernelopsBase : public kernelOpsGeneratorX86<KType>
{
  public:
    explicit kernelopsBase(kernelOpsGeneratorX86<KType>& base)
        : kernelOpsGeneratorX86<KType>(base)
    {
    }

    dlp::jit::jitGeneratorError generate(
        dlp::kernel_frame::kernelOpsMetaData& op)
    {
        return static_cast<Derived*>(this)->generateImpl(op);
    }
};

// ═════════════════════════════════════════════════════════════════════════
// Strategy classes
// ═════════════════════════════════════════════════════════════════════════

// MatOps: C += / *= auxiliary matrix (row-major, col-major, GEMV-N1 paths)
template<utils::kernelInstrType KType>
class MatOps : public kernelopsBase<MatOps<KType>, KType>
{
    using opBase = kernelopsBase<MatOps<KType>, KType>;
    using typename opBase::halfRegType;
    using typename opBase::RegType;
    using typename opBase::Traits;

  public:
    enum class matOpType
    {
        matOpAdd,
        matOpMul
    };
    enum class matOpScaleType
    {
        scalar,
        rowVector,
        columnVector
    };

  private:
    inline void applyMatOp(matOpType            opType,
                           bool                 hasSF,
                           bool                 isFringe,
                           int                  accumIdx,
                           int                  matRegIdx,
                           int                  sfRegIdx,
                           const Xbyak::Opmask& fringeMask);

    utils::registerGuard<Xbyak::Opmask> gatherMask0;
    utils::registerGuard<Xbyak::Opmask> gatherMask1;

    void resetGatherMasks(bool useFringe, int fringeIdx)
    {
        if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
            return;
        }

        if (useFringe && this->numFringeMasks > 0) {
            Xbyak::Opmask fringeMask = this->getFringeMask(fringeIdx);
            if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
                gatherMask0.reset(true, fringeMask, 0);
                gatherMask1.reset(true, fringeMask, 8);
            } else if constexpr (KType
                                 == utils::kernelInstrType::avx512_ymm_32_reg) {
                gatherMask0.reset(true, fringeMask, 0);
                gatherMask1.reset(true, fringeMask, 4);
            }
        } else {
            gatherMask0.resetToAllOnes();
            gatherMask1.resetToAllOnes();
        }
    }

  public:
    explicit MatOps(kernelOpsGeneratorX86<KType>& base)
        : opBase(base)
    {
    }

    dlp::jit::jitGeneratorError generateImpl(
        dlp::kernel_frame::kernelOpsMetaData& op);

  private:
    dlp::jit::jitGeneratorError rowMajorPath(
        matOpType                   opType,
        matOpScaleType              sclType,
        bool                        hasSF,
        dlp::kernel_frame::DataType sfDtype,
        dlp::kernel_frame::DataType matOpDtype,
        int                         matRegIdx,
        int                         sfRegIdx);

    dlp::jit::jitGeneratorError colMajorPath(
        matOpType                   opType,
        matOpScaleType              sclType,
        bool                        hasSF,
        dlp::kernel_frame::DataType sfDtype,
        dlp::kernel_frame::DataType matOpDtype,
        int                         matRegIdx,
        int                         sfRegIdx);

    dlp::jit::jitGeneratorError gemvN1Path(
        matOpType                   opType,
        matOpScaleType              sclType,
        bool                        hasSF,
        dlp::kernel_frame::DataType sfDtype,
        dlp::kernel_frame::DataType matOpDtype,
        int                         matRegIdx,
        int                         sfRegIdx);
};

// GeluErf: GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
template<utils::kernelInstrType KType>
class GeluErf : public kernelopsBase<GeluErf<KType>, KType>
{
    using opBase = kernelopsBase<GeluErf<KType>, KType>;
    using typename opBase::halfRegType;
    using typename opBase::RegType;
    using typename opBase::Traits;

    static constexpr int NUM_SCRATCH_NEEDED = 9;

    static constexpr int dlpgemmErfOff = sizeof(gen::tables::erf_consts);
    static constexpr int erfF32CoeffsOff =
        dlpgemmErfOff + sizeof(gen::tables::dlp_gemm_erf);
    static constexpr int erfF32ConstantsOff =
        erfF32CoeffsOff + sizeof(gen::tables::erf_f32_coeffs_hex);

    void geluErfF32(int reg,
                    int xtanh,
                    int c1,
                    int c2,
                    int x,
                    int r,
                    int r2,
                    int z,
                    int dn,
                    int q);

    void erf(int y, int r, int c1, int c2, int x, int r2, int z, int dn, int q);

    void polyEvalHorner16(int r, int c1, int c2);

    int                                 erfCmpMaskIdx = -1;
    utils::registerGuard<Xbyak::Opmask> erfMaskGuard;

  public:
    explicit GeluErf(kernelOpsGeneratorX86<KType>& base)
        : opBase(base)
    {
        if constexpr (Traits::hasMaskSupport) {
            erfMaskGuard  = this->maskPool->acquireGuard();
            erfCmpMaskIdx = erfMaskGuard.idx();
        }
    }

    dlp::jit::jitGeneratorError generateImpl(
        dlp::kernel_frame::kernelOpsMetaData& op);
};

// Relu: max(0, x)
template<utils::kernelInstrType KType>
class Relu : public kernelopsBase<Relu<KType>, KType>
{
    using opBase = kernelopsBase<Relu<KType>, KType>;
    using typename opBase::RegType;

    static constexpr int NUM_SCRATCH_NEEDED = 1;

  public:
    explicit Relu(kernelOpsGeneratorX86<KType>& base)
        : opBase(base)
    {
    }

    dlp::jit::jitGeneratorError generateImpl(
        dlp::kernel_frame::kernelOpsMetaData& op);
};

// Swish: x * sigmoid(alpha * x)
template<utils::kernelInstrType KType>
class Swish : public kernelopsBase<Swish<KType>, KType>
{
    using opBase = kernelopsBase<Swish<KType>, KType>;
    using typename opBase::RegType;
    using typename opBase::Traits;

    static constexpr int NUM_SCRATCH_NEEDED = 9;

    void swishF32(int reg,
                  int x_tanh,
                  int x,
                  int const1,
                  int const2,
                  int r,
                  int r2,
                  int z,
                  int dn,
                  int q);

    int                                 expCmpMaskIdx = -1;
    utils::registerGuard<Xbyak::Opmask> expMaskGuard;

  public:
    explicit Swish(kernelOpsGeneratorX86<KType>& base)
        : opBase(base)
    {
        if constexpr (Traits::hasMaskSupport) {
            expMaskGuard  = this->maskPool->acquireGuard();
            expCmpMaskIdx = expMaskGuard.idx();
        }
    }

    dlp::jit::jitGeneratorError generateImpl(
        dlp::kernel_frame::kernelOpsMetaData& op);
};

// GeluTanh: GELU via tanh approximation
template<utils::kernelInstrType KType>
class GeluTanh : public kernelopsBase<GeluTanh<KType>, KType>
{
    using opBase = kernelopsBase<GeluTanh<KType>, KType>;
    using typename opBase::RegType;
    using typename opBase::Traits;

    static constexpr int NUM_SCRATCH_NEEDED = 9;

    void geluTanhF32(int reg,
                     int x_tanh,
                     int x,
                     int const1,
                     int const2,
                     int r,
                     int r2,
                     int z,
                     int dn,
                     int q);

    int                                 expCmpMaskIdx = -1;
    utils::registerGuard<Xbyak::Opmask> expMaskGuard;

  public:
    explicit GeluTanh(kernelOpsGeneratorX86<KType>& base)
        : opBase(base)
    {
        if constexpr (Traits::hasMaskSupport) {
            expMaskGuard  = this->maskPool->acquireGuard();
            expCmpMaskIdx = expMaskGuard.idx();
        }
    }

    dlp::jit::jitGeneratorError generateImpl(
        dlp::kernel_frame::kernelOpsMetaData& op);
};

// Tanh: tanh(x) via EXPF
template<utils::kernelInstrType KType>
class Tanh : public kernelopsBase<Tanh<KType>, KType>
{
    using opBase = kernelopsBase<Tanh<KType>, KType>;
    using typename opBase::RegType;
    using typename opBase::Traits;

    static constexpr int NUM_SCRATCH_NEEDED = 9;

    void tanhDef(int reg,
                 int x_tanh,
                 int x,
                 int const1,
                 int const2,
                 int r,
                 int r2,
                 int z,
                 int dn,
                 int q);

    int                                 expCmpMaskIdx = -1;
    utils::registerGuard<Xbyak::Opmask> expMaskGuard;

  public:
    explicit Tanh(kernelOpsGeneratorX86<KType>& base)
        : opBase(base)
    {
        if constexpr (Traits::hasMaskSupport) {
            expMaskGuard  = this->maskPool->acquireGuard();
            expCmpMaskIdx = expMaskGuard.idx();
        }
    }

    dlp::jit::jitGeneratorError generateImpl(
        dlp::kernel_frame::kernelOpsMetaData& op);
};

// Sigmoid: 1 / (1 + exp(-x))
template<utils::kernelInstrType KType>
class Sigmoid : public kernelopsBase<Sigmoid<KType>, KType>
{
    using opBase = kernelopsBase<Sigmoid<KType>, KType>;
    using typename opBase::RegType;
    using typename opBase::Traits;

    static constexpr int NUM_SCRATCH_NEEDED = 9;

    void sigmoidDef(int reg,
                    int x,
                    int const1,
                    int const2,
                    int r,
                    int r2,
                    int z,
                    int dn,
                    int q);

    int                                 expCmpMaskIdx = -1;
    utils::registerGuard<Xbyak::Opmask> expMaskGuard;

  public:
    explicit Sigmoid(kernelOpsGeneratorX86<KType>& base)
        : opBase(base)
    {
        if constexpr (Traits::hasMaskSupport) {
            expMaskGuard  = this->maskPool->acquireGuard();
            expCmpMaskIdx = expMaskGuard.idx();
        }
    }

    dlp::jit::jitGeneratorError generateImpl(
        dlp::kernel_frame::kernelOpsMetaData& op);
};

// Clip: clamp(x, min, max)
template<utils::kernelInstrType KType>
class Clip : public kernelopsBase<Clip<KType>, KType>
{
    using opBase = kernelopsBase<Clip<KType>, KType>;
    using typename opBase::RegType;

    static constexpr int NUM_SCRATCH_NEEDED = 2;

  public:
    explicit Clip(kernelOpsGeneratorX86<KType>& base)
        : opBase(base)
    {
    }

    dlp::jit::jitGeneratorError generateImpl(
        dlp::kernel_frame::kernelOpsMetaData& op);
};

// ReluScale: PReLU -- max(0,x) + alpha * min(0,x)
template<utils::kernelInstrType KType>
class ReluScale : public kernelopsBase<ReluScale<KType>, KType>
{
    using opBase = kernelopsBase<ReluScale<KType>, KType>;
    using typename opBase::RegType;
    using typename opBase::Traits;

    static constexpr int NUM_SCRATCH_NEEDED = Traits::hasMaskSupport ? 2 : 3;

    int                                 cmpMaskIdx = -1;
    utils::registerGuard<Xbyak::Opmask> cmpMaskGuard;

  public:
    explicit ReluScale(kernelOpsGeneratorX86<KType>& base)
        : opBase(base)
    {
        if constexpr (Traits::hasMaskSupport) {
            cmpMaskGuard = this->maskPool->acquireGuard();
            cmpMaskIdx   = cmpMaskGuard.idx();
        }
    }

    dlp::jit::jitGeneratorError generateImpl(
        dlp::kernel_frame::kernelOpsMetaData& op);
};

// Downscale: C = C*SF + ZP (both), C = C*SF (SF-only), or C = C+ZP (ZP-only)
template<utils::kernelInstrType KType>
class Downscale : public kernelopsBase<Downscale<KType>, KType>
{
    using opBase = kernelopsBase<Downscale<KType>, KType>;
    using typename opBase::RegType;
    using typename opBase::Traits;

    static constexpr int NUM_SCRATCH_NEEDED = 3;

    enum class LoadMode
    {
        Scalar,
        PerCol,
        PerRow
    };

    LoadMode getLoadMode(bool                             isScalar,
                         dlp::kernel_frame::storageFormat fmt) const;

    dlp::jit::jitGeneratorError applyDownscale(int      effectiveMR,
                                               bool     hasSF,
                                               bool     hasZP,
                                               LoadMode sfMode,
                                               LoadMode zpMode,
                                               dlp::kernel_frame::DataType sfDt,
                                               dlp::kernel_frame::DataType zpDt,
                                               const Xbyak::Reg64& sfBase,
                                               const Xbyak::Reg64& zpBase);

  public:
    explicit Downscale(kernelOpsGeneratorX86<KType>& base)
        : opBase(base)
    {
    }

    dlp::jit::jitGeneratorError generateImpl(
        dlp::kernel_frame::kernelOpsMetaData& op);
};

// Bias: C += bias (legacy) or C += SF * (bias - ZP) (dequantization)
template<utils::kernelInstrType KType>
class Bias : public kernelopsBase<Bias<KType>, KType>
{
    using opBase = kernelopsBase<Bias<KType>, KType>;
    using typename opBase::RegType;
    using typename opBase::Traits;

    static constexpr int NUM_SCRATCH_NEEDED = 3;

    dlp::jit::jitGeneratorError legacyBiasVector(
        dlp::kernel_frame::DataType biasDt,
        int                         effectiveMR,
        const Xbyak::Reg64&         biasBase);
    dlp::jit::jitGeneratorError legacyBiasBroadcast(
        dlp::kernel_frame::DataType biasDt,
        int                         effectiveMR,
        const Xbyak::Reg64&         biasBase);
    dlp::jit::jitGeneratorError scalarBiasBroadcast(
        dlp::kernel_frame::DataType biasDt,
        int                         effectiveMR,
        const Xbyak::Reg64&         biasBase);

    dlp::jit::jitGeneratorError biasDeQuantPerCol(
        int                         effectiveMR,
        bool                        biasIsBroadcast,
        dlp::kernel_frame::DataType biasDt,
        dlp::kernel_frame::DataType sfDt,
        dlp::kernel_frame::DataType zpDt,
        bool                        hasSF,
        bool                        hasZP,
        bool                        sfIsScalar,
        bool                        zpIsScalar,
        const Xbyak::Reg64&         biasBase,
        const Xbyak::Reg64&         sfBase,
        const Xbyak::Reg64&         zpBase);

    dlp::jit::jitGeneratorError biasDeQuantPerRow(
        int                         effectiveMR,
        bool                        biasIsBroadcast,
        dlp::kernel_frame::DataType biasDt,
        dlp::kernel_frame::DataType sfDt,
        dlp::kernel_frame::DataType zpDt,
        bool                        hasSF,
        bool                        hasZP,
        bool                        sfIsScalar,
        bool                        zpIsScalar,
        const Xbyak::Reg64&         biasBase,
        const Xbyak::Reg64&         sfBase,
        const Xbyak::Reg64&         zpBase);

  public:
    explicit Bias(kernelOpsGeneratorX86<KType>& base)
        : opBase(base)
    {
    }

    dlp::jit::jitGeneratorError generateImpl(
        dlp::kernel_frame::kernelOpsMetaData& op);
};

// ADQuantize: result = (acc + b_col_sum * a_zp) * a_sf
template<utils::kernelInstrType KType>
class ADQuantize : public kernelopsBase<ADQuantize<KType>, KType>
{
    using opBase = kernelopsBase<ADQuantize<KType>, KType>;
    using typename opBase::RegType;
    using typename opBase::Traits;

    static constexpr int NUM_SCRATCH_NEEDED = 3;

    dlp::jit::jitGeneratorError aDQuantScaleFactorImpl(
        dlp::kernel_frame::kernelOpsMetaData& op);
    dlp::jit::jitGeneratorError aDQuantScaleFactorScalarImpl(
        dlp::kernel_frame::DataType sfDt, const Xbyak::Reg64& sfBase);
    dlp::jit::jitGeneratorError aDQuantScaleFactorScalarImplGEMVN1(
        dlp::kernel_frame::DataType sfDt, const Xbyak::Reg64& sfBase);
    dlp::jit::jitGeneratorError aDQuantScaleFactorRowMajorImpl(
        dlp::kernel_frame::DataType sfDt, const Xbyak::Reg64& sfBase);
    dlp::jit::jitGeneratorError aDQuantScaleFactorRowMajorImplGEMVN1(
        dlp::kernel_frame::DataType sfDt, const Xbyak::Reg64& sfBase);

    dlp::jit::jitGeneratorError aDQuantZeroPointImpl(
        dlp::kernel_frame::kernelOpsMetaData& op);
    dlp::jit::jitGeneratorError aDQuantZeroPointScalarImpl(
        dlp::kernel_frame::DataType zpDt, const Xbyak::Reg64& zpBase);
    dlp::jit::jitGeneratorError aDQuantZeroPointScalarImplGEMVN1(
        dlp::kernel_frame::DataType zpDt, const Xbyak::Reg64& zpBase);
    dlp::jit::jitGeneratorError aDQuantZeroPointRowMajorImpl(
        dlp::kernel_frame::DataType zpDt, const Xbyak::Reg64& zpBase);
    dlp::jit::jitGeneratorError aDQuantZeroPointRowMajorImplGEMVN1(
        dlp::kernel_frame::DataType zpDt, const Xbyak::Reg64& zpBase);

  public:
    explicit ADQuantize(kernelOpsGeneratorX86<KType>& base)
        : opBase(base)
    {
    }

    dlp::jit::jitGeneratorError generateImpl(
        dlp::kernel_frame::kernelOpsMetaData& op);
};

// ═════════════════════════════════════════════════════════════════════════
// Factory: dispatch post-op type -> strategy class
// ═════════════════════════════════════════════════════════════════════════

template<utils::kernelInstrType KType>
inline dlp::jit::jitGeneratorError
generateKernelOp(kernelOpsGeneratorX86<KType>&         base,
                 dlp::kernel_frame::kernelOpsMetaData& op)
{
    switch (op.type) {
        case dlp::kernel_frame::kernelOps::bias: {
            Bias<KType> biasImpl(base);
            return biasImpl.generate(op);
        }
        case dlp::kernel_frame::kernelOps::relu: {
            Relu<KType> reluImpl(base);
            return reluImpl.generate(op);
        }
        case dlp::kernel_frame::kernelOps::reluScale: {
            ReluScale<KType> reluScaleImpl(base);
            return reluScaleImpl.generate(op);
        }
        case dlp::kernel_frame::kernelOps::geluTanh: {
            GeluTanh<KType> geluTanhImpl(base);
            return geluTanhImpl.generate(op);
        }
        case dlp::kernel_frame::kernelOps::geluErf: {
            GeluErf<KType> geluErfImpl(base);
            return geluErfImpl.generate(op);
        }
        case dlp::kernel_frame::kernelOps::clip: {
            Clip<KType> clipImpl(base);
            return clipImpl.generate(op);
        }
        case dlp::kernel_frame::kernelOps::downscale: {
            Downscale<KType> downscaleImpl(base);
            return downscaleImpl.generate(op);
        }
        case dlp::kernel_frame::kernelOps::matAdd:
        case dlp::kernel_frame::kernelOps::matMul: {
            MatOps<KType> matOpImpl(base);
            return matOpImpl.generate(op);
        }
        case dlp::kernel_frame::kernelOps::swish: {
            Swish<KType> swishImpl(base);
            return swishImpl.generate(op);
        }
        case dlp::kernel_frame::kernelOps::tanh: {
            Tanh<KType> tanhImpl(base);
            return tanhImpl.generate(op);
        }
        case dlp::kernel_frame::kernelOps::sigmoid: {
            Sigmoid<KType> sigmoidImpl(base);
            return sigmoidImpl.generate(op);
        }
        case dlp::kernel_frame::kernelOps::aDQuantize: {
            ADQuantize<KType> adQuantImpl(base);
            return adQuantImpl.generate(op);
        }
        default:
            return dlp::jit::jitGeneratorError::notSupported;
    }
}

} // namespace amdzen::x86gen
